/**********************************************************************
 * ----  Shear sort ----
 * Usage: mpirun -np p ./shear_sort n m
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
    if (argc < 2)
    {
        printf("ERROR: more input arguments needed\n");
        printf("Usage: mpirun -np p ./shear_sort n m\n");
        return 0;
    }
    MPI_Init(&argc, &argv);
    
    int n = atoi(argv[1]), m = atoi(argv[2]), N = n*m; 
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(n < size){
        printf("  n < p ??? That's wild, and also not allowed. Try again\n");
        MPI_Finalize();
        return 0;
    }

    // heights and number of elements for each PE
    int nCount = n / size, mCount = m / size;
    int nrest = n % size, mrest = m % size, disp = 0, numOfn1 = nrest, numOfm1 = mrest;
    if(numOfn1 == 0) numOfn1 = size;  // quickfix
    if(numOfm1 == 0) numOfm1 = size;  // quickfix

    // Set heights for rows and cols for each PE
    int heights[size], widths[size], displsAll[size], scatterCount[size];
    for (int i = 0; i < size; i++)
    {
        heights[i] = nCount; widths[i] = mCount;
        if (nrest > 0){
            heights[i]++;
            nrest--;
        }
        if (mrest > 0){
            widths[i]++;
            mrest--;
        }
        displsAll[i] = disp;
        disp += m*heights[i];
        scatterCount[i] = m*heights[i];
    }

    // Generate matrix to sort. Uniform distribution between 1 - n^2
    int *big_data = (int*)malloc(N*sizeof(int));
    if (rank == 0){
        srand(time(NULL));
        int max = N;
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                big_data[i * m  + j] = rand() % max; 
            
        // Print to check result if n is reasonably small 
        if(n < 21 && m < 21){
            printf("Initial matrix: \n");
            // Print whole matrix
            for(int i = 0; i < n; i++){
                for(int j = 0; j < m; j++){
                    if(big_data[i * m + j] < 1000) printf(" ");
                    if(big_data[i * m + j] < 100) printf(" ");
                    if(big_data[i * m + j] < 10)  printf(" ");
                    printf("%d ", big_data[i * m + j]);
                }
                printf("\n\n");
            }
            printf("\n\n");
        }
    }

    int *rows = (int*)malloc(m*heights[rank]*sizeof(int));
    int *cols = (int*)malloc(n*widths[rank]*sizeof(int));

    MPI_Datatype blockBig, blockSmall, blockWide, blockLong; // If n % p == 0, only blockBig is used
    MPI_Datatype colType1, colType2; // If n % p == 0, only colType1 is used

    // BLOCK TYPE DEFS
    MPI_Type_vector(heights[0], widths[0], m, MPI_INT, &blockBig);
    MPI_Type_create_resized(blockBig, 0, sizeof(int), &blockBig);
    MPI_Type_commit(&blockBig);

    MPI_Type_vector(heights[0]-1, widths[0]-1, m, MPI_INT, &blockSmall);
    MPI_Type_create_resized(blockSmall, 0, sizeof(int), &blockSmall);
    MPI_Type_commit(&blockSmall);

    MPI_Type_vector(heights[0]-1, widths[0], m, MPI_INT, &blockWide);
    MPI_Type_create_resized(blockWide, 0, sizeof(int), &blockWide);
    MPI_Type_commit(&blockWide);

    MPI_Type_vector(heights[0], widths[0]-1, m, MPI_INT, &blockLong);
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
    int displsRow[size], displsCol[size], sendBlocks[size], sendCols[size];
    MPI_Datatype blockType[size], colType[size];

    int dispRow = 0, dispCol = 0;
    for(int i = 0; i < size; i++){
        displsRow[i] = dispRow;
        displsCol[i] = dispCol;
        sendBlocks[i] = 1;
        if(i < numOfm1){
            dispRow += widths[0]*sizeof(int);
            if(rank < numOfn1){
                blockType[i] = blockBig;
            }else{
                blockType[i] = blockWide; colType[i] = colType2;
            }
        }else{
            dispRow += (widths[0]-1)*sizeof(int);
            if(rank < numOfn1){
                blockType[i] = blockLong; colType[i] = colType1;
            }else{
                blockType[i] = blockSmall; colType[i] = colType2;
            }
        }
        if(i < numOfn1){
            sendCols[i] = heights[0];
            dispCol += heights[0]*sizeof(int);
        }else{
            sendCols[i] = heights[0]-1;
            dispCol += (heights[0]-1)*sizeof(int);
        }
        if(rank < numOfm1){
            colType[i] = colType1;
        }else{
            colType[i] = colType2;
        }
    }
    
    // Find log2(n) with bits and stuff
    int s = n, iter = 0;
    while (s >>= 1) iter++; 
    if((n & (n-1) )!= 0) iter++; // Ceil(log2(n))

    // Set start of even/odd rows
    int evenRowStart = 0;
    if((displsCol[rank]/sizeof(int)) % 2 == 1) evenRowStart++;

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
        for(int i = evenRowStart; i < heights[rank]; i += 2){
            quicksort(&rows[i*m], 0, m-1);
        }

        // Descending order rows
        for(int i = 1 - evenRowStart; i < heights[rank]; i += 2){
            quicksort_descending(&rows[i*m], 0, m-1);
        }

        // Send rows to columns
        MPI_Alltoallw(rows, sendBlocks, displsRow, blockType, cols, sendCols, displsCol, colType, MPI_COMM_WORLD);
        
        // Ascending order cols
        for(int i = 0; i < widths[rank]; i++){
            quicksort(&cols[i*n], 0, n-1);
        }
        // Send columns to rows
        MPI_Alltoallw(cols, sendCols, displsCol, colType, rows, sendBlocks, displsRow, blockType, MPI_COMM_WORLD);
    }
    // Ascending order rows
    for(int i = evenRowStart; i < heights[rank]; i += 2){
        quicksort(&rows[i*m], 0, m-1);
    }

    // Descending order rows
    for(int i = 1 - evenRowStart; i < heights[rank]; i += 2){
        quicksort_descending(&rows[i*m], 0, m-1);
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
    if(rank == 0 && n < 21 && m < 21){
        printf("Sorted matrix:\n");
        for(int i = 0; i < n; i++){
            for(int j = 0; j < m; j++){
                if(big_data[i * m + j] < 1000) printf(" ");
                if(big_data[i * m + j] < 100) printf(" ");
                if(big_data[i * m + j] < 10)  printf(" ");
                printf("%d ", big_data[i * m + j]);
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

