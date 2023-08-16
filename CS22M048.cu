#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <bits/stdc++.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************
// This kernel counts the number of conflicting requests in a facility ans store it in d_confreqs array
__global__ void countkernell(int* d_req_cen,int* d_req_fac,int* d_confreqs,int R){
      int id=blockIdx.x*1024+threadIdx.x*32+threadIdx.y;
      if(id<R){
        int offset=d_req_cen[id]*max_P+d_req_fac[id];
        atomicAdd(d_confreqs+offset,1);
      }
}
// This kernel sets the capacity value for all slots of each facility in d_slotspercenfac array
__global__ void setarraykernel(int* d_slotspercenfac,int* d_capacity,int N){
    int id=blockIdx.x*1024+threadIdx.x*32+threadIdx.y;
    if(id<N*max_P*24){
        int cen=id/(max_P*24);
        int fac=(id%(max_P*24))/24;
        d_slotspercenfac[id]=d_capacity[cen*max_P+fac];
    }
}
// This kernel processes requests on the  basis of request id for same facility and centre
__global__ void processRequests(int R,  int* req_cen, int* req_fac, int* req_start,
 int* req_slots, int* d_prefix, int* capacity, int* succ_reqs, int* tot_reqs,int *d_succ_cenreqs, int N)
{
    int tid=blockIdx.x*1024+threadIdx.x*32+threadIdx.y;
    // process requests if its a valid facility
    if(tid < N*max_P)
    {
    
        int startreq,endreq;   // startreq, endreq stores the start, end of number of request for that facility
        // Extract request details
        int cen_val = tid/max_P;
        int fac_val = tid%max_P;
        int uniqid = cen_val * max_P + fac_val;    // uniqid stores unique index of facility for this request
        int slot_idx = uniqid*24;      // slot_idx stores the starting index of capacity array for that request
        int slots_available;
        if(uniqid == 0)      // if uniqid is 0 startreq will be zero and the previos prefix array value otherwise
            startreq = 0;
        else
            startreq=d_prefix[uniqid-1];

        endreq=d_prefix[uniqid];
        for(int i=startreq;i<endreq;i++){       //process request for this facility
            slots_available=0;
            int slot_start_val = req_start[i];
            int req_slots_val = req_slots[i];
            // Check if requested slots are available
            for (int i = slot_start_val-1; i < slot_start_val + req_slots_val -1 ; i++)
            {
                //if slot is available count it
                if (capacity[slot_idx + i] > 0)
                {
                    slots_available++;
                }
            }
            // If all requested slots are available, mark request as successful
            if (slots_available == req_slots_val)
            {
                // Decrement capacity for the granted request
                for (int i = slot_start_val-1; i < slot_start_val + req_slots_val -1; i++)
                {
                    capacity[slot_idx + i]--;
                }
                atomicAdd(&d_succ_cenreqs[cen_val], 1);    //atomically increment succesfull request for this centre
                atomicAdd(succ_reqs,1);       //atomically increment succesfull request in all
            }
            atomicAdd(&tot_reqs[cen_val], 1);      //atomically increment this in number of requests
        }
    }
}
int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*d_capacity, *fac_ids, *succ_cenreqs, *tot_reqs;
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 
    memset(capacity,0,max_P*N*sizeof(int));
    cudaMalloc(&d_capacity, N*max_P*sizeof(int));   ///allocating memory for capacity array in GPU
    
    int success,fail;  // total successful,failed requests
    int *succ_reqs=(int*)malloc(sizeof (int));  //variable in CPU
    int *d_succ_reqs;
    cudaMalloc(&d_succ_reqs, sizeof(int));     //variable in GPU
    cudaMemset(d_succ_reqs,0,sizeof(int));

    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_cenreqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre
    int* d_tot_reqs,*d_succ_cenreqs;   //device arrays
    cudaMalloc(&d_tot_reqs, N*sizeof(int));
    cudaMemset(d_tot_reqs,0,sizeof(int));
    cudaMalloc(&d_succ_cenreqs, N*sizeof(int));
    cudaMemset(d_succ_cenreqs,0,sizeof(int));

    // Input the computer centres data
    int k1=0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[centre[i]*max_P+j]);  //setting capacity of each facility for each centre in capacity array
      }
    }
    cudaMemcpy(d_capacity, capacity, N*max_P*sizeof(int), cudaMemcpyHostToDevice);  //setting value in d_capacity array
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots,*d_req_cen,*d_req_fac;   // Number of slots requested for every request
    int *d_confreqs,*confreqs, *d_prefix, *prefix, *d_slotspercenfac; //d_prefix: device arr which counts the starting index of requests to same computer centre and facility room number
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    confreqs = (int *) malloc ( (N * max_P) * sizeof (int) ); // counts request to same facility
    prefix = (int *) malloc ( (N * max_P) * sizeof (int) );   // stores the starting index of requests on a particular facility

    //Allocate memory on GPU
    cudaMalloc(&d_slotspercenfac, N*max_P*24*sizeof(int));
    cudaMalloc(&d_confreqs, N*max_P*sizeof(int));
    cudaMalloc(&d_prefix, N*max_P*sizeof(int));
    cudaMalloc(&d_req_cen, R*sizeof(int));
    cudaMalloc(&d_req_fac, R*sizeof(int));
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
    //set and copy values into the d_array
    cudaMemcpy(d_req_cen, req_cen, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac, req_fac, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_confreqs, 0, N*max_P*sizeof(int));
    cudaMemset(d_prefix, 0, N*max_P*sizeof(int));
    cudaMemset(d_slotspercenfac, 0, N*max_P*24*sizeof(int));
	// Create an index array to keep track of the original indices
    int index[R];
    for (int i = 0; i < R; ++i) {
        index[i] = i;
    }
    // sort this index array according to centre id then facility id and then request id
    std::sort(index, index + R, [&](int a, int b) {
        if (req_cen[a] != req_cen[b]) {
            return req_cen[a] < req_cen[b];
        } else if (req_fac[a] != req_fac[b]) {
            return req_fac[a] < req_fac[b];
        } else {
            return req_id[a] < req_id[b];
        }
    });
    //sorted request variables
    int *req_cen_sorted,*req_fac_sorted,*req_start_sorted,*req_slots_sorted,*req_id_sorted;
    req_cen_sorted = (int *) malloc ( (R) * sizeof (int) );
    req_fac_sorted = (int *) malloc ( (R) * sizeof (int) );
    req_start_sorted = (int *) malloc ( (R) * sizeof (int) );
    req_slots_sorted = (int *) malloc ( (R) * sizeof (int) );
    req_id_sorted = (int *) malloc ( (R) * sizeof (int) );
    // Rearrange the input arrays based on the sorted indices
    for (int i = 0; i < R; ++i)
    {
        int idx = index[i];
        req_cen_sorted[i] = req_cen[idx];
        req_fac_sorted[i] = req_fac[idx];
        req_start_sorted[i] = req_start[idx];
        req_slots_sorted[i] = req_slots[idx];
        req_id_sorted[i] = req_id[idx];
    }
    //device variables after arrays are sorted
    int *d_req_cen_sorted, *d_req_fac_sorted, *d_req_start_sorted, *d_req_slots_sorted;
    cudaMalloc(&d_req_cen_sorted, R*sizeof(int));
    cudaMalloc(&d_req_fac_sorted, R*sizeof(int));
    cudaMalloc(&d_req_start_sorted, R*sizeof(int));
    cudaMalloc(&d_req_slots_sorted, R*sizeof(int));
    cudaMemcpy(d_req_cen_sorted, req_cen_sorted, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_fac_sorted, req_fac_sorted, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_start_sorted, req_start_sorted, R*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots_sorted, req_slots_sorted, R*sizeof(int), cudaMemcpyHostToDevice);

    //*********************************
    // Call the kernels here
    //********************************

    dim3 gridDim(ceil(float(R)/1024),1,1);  
    dim3 blockDim(32,32,1);
    //count the req for same cen and fac
    countkernell<<<gridDim,blockDim>>>(d_req_cen,d_req_fac,d_confreqs,R);
    cudaMemcpy(confreqs,d_confreqs,N*max_P*sizeof(int),cudaMemcpyDeviceToHost);

    //free d_req_cen and d_req_fac
    cudaFree(d_req_cen);
    cudaFree(d_req_fac);

    //calculate prefix sum
    prefix[0] = confreqs[0]; // First element of prefix sum array is same as original array
    for (int i = 1; i < N*max_P; i++) {
        prefix[i] = prefix[i - 1] + confreqs[i]; // Calculate prefix sum
    }
    cudaMemcpy(d_prefix, prefix, N*max_P * sizeof(int), cudaMemcpyHostToDevice);  //store prefix in GPU

    //set the values of d_slotpercenfac in GPU
    dim3 gridDim1(ceil(float(N*max_P*24)/1024),1,1);  
    dim3 blockDim1(32,32,1);
    setarraykernel<<<gridDim1,blockDim1>>>(d_slotspercenfac,d_capacity,N);  //d_capacity stores the capacity of each facility

    //launching successfull request counting kernel with "total possible facility" number of threads
    dim3 gridDim2(ceil(float(N*max_P)/1024),1,1);  
    dim3 blockDim2(32,32,1);
    processRequests<<<gridDim2,blockDim2>>>(R,d_req_cen_sorted,d_req_fac_sorted,d_req_start_sorted,d_req_slots_sorted,d_prefix,d_slotspercenfac,d_succ_reqs,d_tot_reqs,d_succ_cenreqs,N);
    // storeback variables in CPU
    cudaMemcpy(succ_reqs, d_succ_reqs,sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tot_reqs, d_tot_reqs, N* sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(succ_cenreqs, d_succ_cenreqs, N* sizeof(int), cudaMemcpyDeviceToHost);

    success=*succ_reqs;  //storing total successfull requests
    fail=R-success;      //storing total failed requests
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);

    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_cenreqs[j], tot_reqs[j]-succ_cenreqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}