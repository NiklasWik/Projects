/**********************************************************************
 * ----  Shear sort ----
 * Usage: mpirun -np p ./shear_sort n
 * Niklas Wik 
 **********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

int partition(int *data, int left, int right, int pivotIndex);
void quicksort(int *data, int left, int right);
int partition_descending(int *data, int left, int right, int pivotIndex);
void quicksort_descending(int *data, int left, int right);

int main(int argc, char *argv[])
{
    if (argc < 1)
    {
        printf("ERROR: more input arguments needed\n");
        printf("Usage: mpirun -np p ./shear_sort n\n");
        return 0;
    }
    MPI_Init(&argc, &argv);
    
    int n = atoi(argv[1]), N = n*n; 
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(n < size){
        printf("  n < p ??? That's wild, and also not allowed. Try again\n");
        MPI_Finalize();
        return 0;
    }

    // Widths and number of elements for each PE
    int count = n / size;
    int rest = n % size, disp = 0, numOfw1 = rest;
    if(numOfw1 == 0) numOfw1 = size;  // quickfix

    // Set widths for rows and cols for each PE
    int widths[size], displsAll[size], scatterCount[size];
    for (int i = 0; i < size; i++)
    {
        widths[i] = count;
        if (rest > 0)
        {
            widths[i]++;
            rest--;
        }
        displsAll[i] = disp;
        disp += n*widths[i];
        scatterCount[i] = n*widths[i];
    }

    // Generate matrix to sort. Uniform distribution between 1 - n^2
    int *big_data = (int*)malloc(N*sizeof(int));
    if (rank == 0){
        srand(time(NULL));
        int max = 4*N;
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                big_data[i * n  + j] = rand() % max; 
            
        // Print to check result if n is reasonably small 
        if(n < 21){
            printf("Initial matrix: \n");
            // Print whole matrix
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    if(big_data[i * n + j] < 1000) printf(" ");
                    if(big_data[i * n + j] < 100) printf(" ");
                    if(big_data[i * n + j] < 10)  printf(" ");
                    printf("%d ", big_data[i * n + j]);
                }
                printf("\n\n");
            }
            printf("\n\n");
        }
    }

    int *rows = (int*)malloc(n*widths[rank]*sizeof(int));
    int *cols = (int*)malloc(n*widths[rank]*sizeof(int));

    MPI_Datatype blockBig, blockSmall, blockWide, blockLong; // If n % p == 0, only blockBig is used
    MPI_Datatype colType1, colType2; // If n % p == 0, only colType1 is used

    // BLOCK TYPE DEFS
    MPI_Type_vector(widths[0], widths[0], n, MPI_INT, &blockBig);
    MPI_Type_create_resized(blockBig, 0, sizeof(int), &blockBig);
    MPI_Type_commit(&blockBig);

    MPI_Type_vector(widths[0]-1, widths[0]-1, n, MPI_INT, &blockSmall);
    MPI_Type_create_resized(blockSmall, 0, sizeof(int), &blockSmall);
    MPI_Type_commit(&blockSmall);

    MPI_Type_vector(widths[0]-1, widths[0], n, MPI_INT, &blockWide);
    MPI_Type_create_resized(blockWide, 0, sizeof(int), &blockWide);
    MPI_Type_commit(&blockWide);

    MPI_Type_vector(widths[0], widths[0]-1, n, MPI_INT, &blockLong);
    MPI_Type_create_resized(blockLong, 0, sizeof(int), &blockLong);
    MPI_Type_commit(&blockLong);

    // COLUMN TYPE DEF
    MPI_Type_vector(widths[0], 1, n, MPI_INT, &colType1);
    MPI_Type_create_resized(colType1, 0, sizeof(int), &colType1);
    MPI_Type_commit(&colType1);

    MPI_Type_vector(widths[0]-1, 1, n, MPI_INT, &colType2);
    MPI_Type_create_resized(colType2, 0, sizeof(int), &colType2);
    MPI_Type_commit(&colType2);


    // Set different blocks and columns to send/recv. 
    int displs[size], sendBlocks[size], sendCols[size];
    MPI_Datatype blockType[size], colType[size];

    disp = 0; 
    for(int i = 0; i < size; i++){
        displs[i] = disp;
        sendBlocks[i] = 1;
        if(i < numOfw1){
            disp += widths[0]*sizeof(int);
            sendCols[i] = widths[0];
            if(rank < numOfw1){
                blockType[i] = blockBig; colType[i] = colType1;
            }else{
                blockType[i] = blockWide; colType[i] = colType2;
            }
        }else{
            disp += (widths[0]-1)*sizeof(int);
            sendCols[i] = widths[0]-1;
            if(rank < numOfw1){
                blockType[i] = blockLong; colType[i] = colType1;
            }else{
                blockType[i] = blockSmall; colType[i] = colType2;
            }
        }
    }
    
    // Find log2(n) with bits and stuff
    int s = n, iter = 0;
    while (s >>= 1) iter++; 
    if((n & (n-1) )!= 0) iter++; // Ceil(log2(n))

    // Set start of even/odd rows
    int evenRowStart = 0;
    if((displs[rank]/sizeof(int)) % 2 == 1) evenRowStart++;

    // Synchronize PE:s before starting timers
    MPI_Barrier(MPI_COMM_WORLD);

    // Start timer 
    double start = MPI_Wtime();

    // SHEAR SORT BEGINS

    // Distribute matrix to PE:s
    MPI_Scatterv(big_data, scatterCount, displsAll, MPI_INT, rows, scatterCount[rank],
                 MPI_INT, 0, MPI_COMM_WORLD);  
    
    // Free big matrix
    free(big_data);

    for(int k = 0; k < iter; k++){
        // Ascending order rows
        for(int i = evenRowStart; i < widths[rank]; i += 2){
            quicksort(&rows[i*n], 0, n-1);
        }

        // Descending order rows
        for(int i = 1 - evenRowStart; i < widths[rank]; i += 2){
            quicksort_descending(&rows[i*n], 0, n-1);
        }

        // Send rows to columns
        MPI_Alltoallw(rows, sendBlocks, displs, blockType, cols, sendCols, displs, colType, MPI_COMM_WORLD);
        
        // Ascending order cols
        for(int i = 0; i < widths[rank]; i++){
            quicksort(&cols[i*n], 0, n-1);
        }
        // Send columns to rows
        MPI_Alltoallw(cols, sendCols, displs, colType, rows, sendBlocks, displs, blockType, MPI_COMM_WORLD);
    }
    // Ascending order rows
    for(int i = evenRowStart; i < widths[rank]; i += 2){
        quicksort(&rows[i*n], 0, n-1);
    }

    // Descending order rows
    for(int i = 1 - evenRowStart; i < widths[rank]; i += 2){
        quicksort_descending(&rows[i*n], 0, n-1);
    }

    // Gather all data into one matrix
    big_data = (int*)malloc(N*sizeof(int));

    MPI_Gatherv(rows, scatterCount[rank], MPI_INT, big_data, scatterCount, displsAll,
                 MPI_INT, 0, MPI_COMM_WORLD);  

    // Stop timer! 
    double ex_time = MPI_Wtime() - start;
    double *timings = (double*)malloc(size*sizeof(double));
    MPI_Gather(&ex_time, 1, MPI_DOUBLE, timings, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Find longest time
    if(rank == 0){
        for(int i = 1; i < size; i++){
            if(ex_time < timings[i])
                ex_time = timings[i];
        }
        printf("\n\nTime: %lf\n\n", ex_time);
    }

    // Print to check result if reasonably small 
    if(rank == 0 && n < 21){
        printf("Sorted matrix:\n");
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                if(big_data[i * n + j] < 1000) printf(" ");
                if(big_data[i * n + j] < 100) printf(" ");
                if(big_data[i * n + j] < 10)  printf(" ");
                printf("%d ", big_data[i * n + j]);
            }
            printf("\n\n");
        }
        printf("\n\n");
    }

    MPI_Finalize();
    return 0;
}

int partition(int *data, int left, int right, int pivotIndex)
{
    int pivotValue, temp;
    int storeIndex, i;
    pivotValue = data[pivotIndex];
    temp = data[pivotIndex];
    data[pivotIndex] = data[right];
    data[right] = temp;
    storeIndex = left;
    for (i = left; i < right; i++)
        if (data[i] < pivotValue)
        {
            temp = data[i];
            data[i] = data[storeIndex];
            data[storeIndex] = temp;
            storeIndex = storeIndex + 1;
        }
    temp = data[storeIndex];
    data[storeIndex] = data[right];
    data[right] = temp;
    return storeIndex;
}

void quicksort(int *data, int left, int right)
{
    int pivotIndex, pivotNewIndex;

    if (right > left)
    {
        pivotIndex = left + (right - left) / 2;
        pivotNewIndex = partition(data, left, right, pivotIndex);
        quicksort(data, left, pivotNewIndex - 1);
        quicksort(data, pivotNewIndex + 1, right);
    }
}

int partition_descending(int *data, int left, int right, int pivotIndex)
{
    int pivotValue, temp;
    int storeIndex, i;
    pivotValue = data[pivotIndex];
    temp = data[pivotIndex];
    data[pivotIndex] = data[right];
    data[right] = temp;
    storeIndex = left;
    for (i = left; i < right; i++)
        if (data[i] > pivotValue)
        {
            temp = data[i];
            data[i] = data[storeIndex];
            data[storeIndex] = temp;
            storeIndex = storeIndex + 1;
        }
    temp = data[storeIndex];
    data[storeIndex] = data[right];
    data[right] = temp;
    return storeIndex;
}

void quicksort_descending(int *data, int left, int right)
{
    int pivotIndex, pivotNewIndex;

    if (right > left)
    {
        pivotIndex = left + (right - left) / 2;
        pivotNewIndex = partition_descending(data, left, right, pivotIndex);
        quicksort_descending(data, left, pivotNewIndex - 1);
        quicksort_descending(data, pivotNewIndex + 1, right);
    }
}

