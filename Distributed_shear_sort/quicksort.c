/**********************************************************************
 * Quick sort using MPI. Niklas Wik & David Niemelä 
 * Usage: ./quicksort sequence length pivot-strat
 *
 **********************************************************************/
#define PI 3.14159265358979323846
#define _XOPEN_SOURCE // drand48 undeclared
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


int partition(double *data, int left, int right, int pivotIndex);
void quicksort(double *data, int left, int right);
double *merge(double *v1, int n1, double *v2, int n2);
double pivotMedian(double *data, int n, MPI_Comm comm);
double pivotMedianOfMedian(double *data, int n, MPI_Comm comm);
double pivotMeanOfMedian(double *data, int n, MPI_Comm comm);
void globSort(double *data, int n, int iter, MPI_Comm comm);
double (*pivotFuncs[3])(double *, int, MPI_Comm) =
         {&pivotMedian, &pivotMedianOfMedian, &pivotMeanOfMedian};


// Global variables
int size, wrank, len, seq, pivStrat, k, totN;
double (*pivotFunc)(double *, int, MPI_Comm);
MPI_Status status;
MPI_Comm cubeComm;
double *allData;
int *destination;
int *coords;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("ERROR: more input arguments needed\n");
        printf("./quicksort sequence length pivot-strategy\n");
        return 0;
    }

    MPI_Init(&argc, &argv);

    seq = atoi(argv[1]); len = atoi(argv[2]); pivStrat = atoi(argv[3]);

    if(pivStrat > 2 || pivStrat < 0){
        printf("Invalid pivot strategy choose 0, 1 or 2.\n");
        MPI_Finalize();
        return 0;
    }
    if(seq > 3 || seq < 0){
        printf("Invalid sequence strategy choose 0, 1 or 2.\n");
        MPI_Finalize();
        return 0;
    }
    if(len < 1){
        printf("Invalid length.\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if ((size & (size - 1)) != 0)
    {
        printf("Number of processors need to be a power of 2\n");
        MPI_Finalize();
        return 0;
    }

    // Find k, #PE = 2^k 
    k = 0;
    int s = size;
    while (s >>= 1)
    {
        k++; 
    }

    // Create hypercube communicator
    int dims[k], period[k];
    coords = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
    {
        dims[i] = 2;
        period[i] = 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Cart_create(MPI_COMM_WORLD, k, dims, period, 1, &cubeComm);
    MPI_Cart_coords(cubeComm, wrank, k, coords);

    // Find all neighbours 
    int source[k];
    destination = (int *)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++)
    {
        MPI_Cart_shift(cubeComm, i, 1, &source[i], &destination[i]);
    }

    // Generate data to sort
    allData = (double *)malloc(len * sizeof(double));
    if (wrank == 0)
    {
        if (seq == 0)
        {
            // Uniform random numbers
            for (int i = 0; i < len; i++)
            {
                allData[i] = drand48();
            }
        }

        else if (seq == 1)
        {
            // Exponential distribution
            double lambda = 10;
            for (int i = 0; i < len; i++)
                allData[i] = -lambda * log(1 - drand48());
        }

        else if (seq == 2)
        {
            // Normal distribution
            double x, y;
            for (int i = 0; i < len; i++)
            {
                x = drand48();
                y = drand48();
                allData[i] = sqrt(-2 * log(x)) * cos(2 * PI * y);
            }
        }
        
        else if (seq == 3)
        {
            double temp = len*1.0;
            // Descending order, fully unsorted
            for (int i = 0; i < len; i++)
            {
                allData[i] = temp--;
            }
        }
    }
    
    // Displacement and number of elements for each PE
    int count = len / size;
    int rest = len % size, disp = 0;
    int sendcounts[size], displs[size];
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = count;
        if (rest > 0)
        {
            sendcounts[i]++;
            rest--;
        }
        displs[i] = disp;
        disp += sendcounts[i];
    }
    double *pdata = (double *)malloc(sendcounts[wrank] * sizeof(double));

    // Pivstrat is choosen 
    pivotFunc = pivotFuncs[pivStrat];

    // Start timer after data is generated 
    MPI_Barrier(cubeComm);

    // Start timer! 
    double start = MPI_Wtime();

    MPI_Scatterv(allData, sendcounts, displs, MPI_DOUBLE, pdata, sendcounts[wrank],
                 MPI_DOUBLE, 0, cubeComm);  
    
    // Local sort!
    quicksort(pdata, 0, sendcounts[wrank] - 1);

    // Global sorting!
    globSort(pdata, sendcounts[wrank], 0, cubeComm);

    // Stop timer! 
    double ex_time = MPI_Wtime() - start;

    double *timings = (double*)malloc(size*sizeof(double));

    MPI_Gather(&ex_time, 1, MPI_DOUBLE, timings, 1, MPI_DOUBLE, 0, cubeComm);
    // Find longest time
    if(wrank == 0){
        for(int i = 1; i < size; i++){
            if(ex_time < timings[i])
                ex_time = timings[i];
        }
        printf("\n\nTime: %lf\n\n", ex_time);
    }
 
    // Check results
    if (wrank == 0)
    {
        int OK = 1;
        for (int i = 0; i < len - 1; i++)
        {
            if (allData[i] > allData[i + 1])
            {
                printf("Wrong result: data[%d] = %lf, data[%d] = %lf\n", i, allData[i], i + 1, allData[i + 1]);
                OK = 0;
            }
        }
        if (OK)
            printf("Data sorted correctly!\n");
    }
    free(allData);
    free(timings);
    MPI_Comm_free(&cubeComm);
    MPI_Finalize();
    return 0;
}

int partition(double *data, int left, int right, int pivotIndex)
{
    double pivotValue, temp;
    int storeIndex, i;
    pivotValue = data[pivotIndex];
    temp = data[pivotIndex];
    data[pivotIndex] = data[right];
    data[right] = temp;
    storeIndex = left;
    for (i = left; i < right; i++)
        if (data[i] <= pivotValue)
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

void quicksort(double *data, int left, int right)
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

inline double pivotMedian(double *data, int n, MPI_Comm comm)
{
    double pivotValue;
    int index = n / 2;

    pivotValue = data[index];
    return pivotValue;
}

inline double pivotMedianOfMedian(double *data, int n, MPI_Comm comm)
{
    double pivotValue;
    int num, index = n / 2;
    MPI_Comm_size(comm, &num);
    double *medians = (double *)malloc(num * sizeof(double));

    pivotValue = data[index];

    MPI_Gather(&pivotValue, 1, MPI_DOUBLE, medians, 1, MPI_DOUBLE, 0, comm);

    // Sort medians with insertion sort. Maybe use quicksort for large #PE
    int i = 1, j;
    double temp;
    while (i < num){
        j = i;
        while (j > 0 && medians[j-1] > medians[j]){
            temp = medians[j];
            medians[j] = medians[j-1];
            medians[j-1] = temp;
            j--;
        }
        i++;
    }

    // Median of medians 
    index = num / 2;
    pivotValue = medians[index];
    free(medians);
    return pivotValue;
}

inline double pivotMeanOfMedian(double *data, int n, MPI_Comm comm)
{
    double pivotValue, sum;
    int num, index = n / 2;
    MPI_Comm_size(comm, &num);
    pivotValue = data[index];
    MPI_Reduce(&pivotValue, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    
    pivotValue = sum/num;
    return pivotValue;
}

void globSort(double *data, int n, int iter, MPI_Comm comm)
{
    int index = n / 2, index2;
    double pivotValue = (*pivotFunc)(data, n, comm);

    MPI_Bcast(&pivotValue, 1, MPI_DOUBLE, 0, comm);

    // Find from what index to send
    while (index > 0 && data[index] > pivotValue)
    {
        index--;
    }
    while (index < n && data[index] < pivotValue)
    {
        index++;
    }

    // Pair up PE:s
    int reccount, sendcount;
    if (wrank < destination[iter])
    {
        sendcount = (n - index);
        index2 = 0;
    }
    else
    {
        sendcount = index;
        index2 = index;
        index = 0;
    }

    // Send how many to malloc
    MPI_Sendrecv(&sendcount, 1, MPI_INT, destination[iter], 111,
                 &reccount, 1, MPI_INT, destination[iter], 111, cubeComm, &status);
    double *recvData = (double *)malloc(reccount * sizeof(double));

    // Send part of list
    MPI_Sendrecv(&data[index], sendcount, MPI_DOUBLE, destination[iter], 222,
                 recvData, reccount, MPI_DOUBLE, destination[iter], 222, cubeComm, &status);

    double *mData = merge(&data[index2], n - sendcount, recvData, reccount);

    free(data);
    free(recvData);

    // Check if another iteration is needed
    if (iter + 1 < k)
    {
        MPI_Comm newComm;
        MPI_Comm_split(comm, coords[iter], wrank, &newComm);

        globSort(mData, n - sendcount + reccount, iter + 1, newComm);
    }
    else
    {
        // Gather data in PE:0
        int sendcounts[size], displ[size], disp = 0, N = n - sendcount + reccount;
        MPI_Gather(&N, 1, MPI_INT, sendcounts, 1, MPI_INT, 0, cubeComm);
        for (int i = 0; i < size; i++)
        {
            displ[i] = disp;
            disp += sendcounts[i];
        }
        MPI_Gatherv(mData, N, MPI_DOUBLE, allData, sendcounts, displ, MPI_DOUBLE, 0, cubeComm);
    }
    // Free comm if it isn't cubeComm!
    if(iter > 0) MPI_Comm_free(&comm);
}

double *merge(double *v1, int n1, double *v2, int n2)
{
    int i, j, k;
    double *result;
    result = (double *)malloc((n1 + n2) * sizeof(double));

    i = 0;
    j = 0;
    k = 0;
    while (i < n1 && j < n2)
        if (v1[i] < v2[j])
        {
            result[k] = v1[i];
            i++;
            k++;
        }
        else
        {
            result[k] = v2[j];
            j++;
            k++;
        }
    if (i == n1)
        while (j < n2)
        {
            result[k] = v2[j];
            j++;
            k++;
        }
    else
        while (i < n1)
        {
            result[k] = v1[i];
            i++;
            k++;
        }
    return result;
}
