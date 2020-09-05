
#include<OpenCL/opencl.h>
#include<mex.h>
#include<matrix.h>
//#include"SP.cl.h"
//#include"SP2.cl.h"
#define MAX_BINARY_SIZE (0x100000)
void mexFunction(
        int          nlhs,
        mxArray      *plhs[],
        int          nrhs,
        const mxArray *prhs[]
        )
{
    float dx,dy,k,m,t,U,nx,ny,n,dt;
    int i=1;       
    
    dx =  (cl_float)mxGetScalar(prhs[0]);
    dy =  (cl_float)mxGetScalar(prhs[1]);
    nx =  (cl_int)mxGetM(prhs[8]);
    ny =  (int)mxGetN(prhs[8]);
    t = (cl_float)mxGetScalar(prhs[2]);
    dt = (cl_float)mxGetScalar(prhs[3]);
    k =(cl_float) mxGetScalar(prhs[4]);
    m = (cl_float)mxGetScalar(prhs[5]);
    U = (cl_float)mxGetScalar(prhs[7]);
    n = (cl_float)mxGetScalar(prhs[6]);
    
    int kloc = (int)mxGetScalar(prhs[10]);
    
    
    cl_int nn=(cl_int)ny*(cl_int)nx;    
    cl_double *Zo = (cl_double*)mxGetPr(prhs[8]);
    cl_double* Ao = (cl_double*)mxGetPr(prhs[9]);
    
    cl_float*A = malloc(sizeof(cl_float)*(cl_float)nn);
    cl_float*Z = malloc(sizeof(cl_float)*(cl_float)nn);
    
     cl_float *Zi = malloc(sizeof(cl_float)*nn);
     for (i=0;i<nn;i++)
     {
         Z[i]=(cl_float)Zo[i];
         A[i]=(cl_float)Ao[i];
         Zi[i]=(cl_float)Z[i];
         
     }
     
     
    cl_float ps[7];
    ps[0]=dx;ps[1]=dt;ps[2]=m;ps[3]=n;ps[4]=k;ps[5]=U;ps[6]=(float)nn;
            
    cl_device_id device_ids[16];
    cl_uint num_devices;
    cl_int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 
            sizeof(device_ids), device_ids, &num_devices);
    cl_device_id device_id = device_ids[0];
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue queue=clCreateCommandQueue(context, device_id, 0, 0);
    cl_mem a = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float)*nn,A,NULL);
    cl_mem z = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float)*nn,Z,NULL);
    cl_mem z2 = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float)*nn,Z,NULL);
    cl_mem zi = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float)*nn,Zi,NULL);
    cl_mem pc = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            sizeof(cl_float)*7,&ps,NULL);


    size_t binary_size;
    FILE *fp;

    char fname[]="/Users/gregor/Documents/MATLAB/Matscape3.0/SP.cl";
    fp = fopen(fname,"r");
    
    char* source_str = (char*)malloc(MAX_BINARY_SIZE);
    size_t source_size=fread(source_str,1,MAX_BINARY_SIZE,fp);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
        (const size_t *)&source_size, NULL);
    
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_kernel SP_k=clCreateKernel(program,"SP",NULL);
     
    clSetKernelArg(SP_k,1,sizeof(cl_mem),(void*)&z);

     clSetKernelArg(SP_k,2,sizeof(cl_mem),(void*)&a);
     clSetKernelArg(SP_k,3,sizeof(cl_mem),(void*)&zi);
     clSetKernelArg(SP_k,0,sizeof(cl_mem),(void*)&pc);
     clSetKernelArg(SP_k,4,sizeof(cl_mem),(void*)&z2);
     
     
     /////////////
     

    
    cl_kernel SP2_k=clCreateKernel(program,"SP2",NULL);
    clSetKernelArg(SP2_k,1,sizeof(cl_mem),(void*)&z);
    
    clSetKernelArg(SP2_k,1,sizeof(cl_mem),(void*)&z);

     clSetKernelArg(SP2_k,2,sizeof(cl_mem),(void*)&a);
     clSetKernelArg(SP2_k,3,sizeof(cl_mem),(void*)&zi);
     clSetKernelArg(SP2_k,0,sizeof(cl_mem),(void*)&pc);
          clSetKernelArg(SP2_k,4,sizeof(cl_mem),(void*)&z2);
          
   cl_kernel copi_k=clCreateKernel(program,"copi",NULL);

  clSetKernelArg(copi_k,0,sizeof(cl_mem),(void*)&z);
  clSetKernelArg(copi_k,1,sizeof(cl_mem),(void*)&zi);
  


     ///////////////////////////

     
     
         size_t global_item_size=nn;
         size_t local_item_size=1;
    cl_event event1=NULL;
    cl_event event2=NULL;
    cl_event event3=NULL;
    float ft=t;
    float dft=dt;
    if(t<0)
    {
        dft=-dft;
        ft=-ft;
       
    }

     for (float ts=0;ts<=ft;ts+=dft)
     {
         
     clEnqueueNDRangeKernel(queue, SP_k, 1, NULL,
	&global_item_size, &local_item_size, 0, NULL, &event1);


     clEnqueueNDRangeKernel(queue, SP2_k, 1, NULL,
	&global_item_size, &local_item_size, 1, &event1, &event2);
     
     clEnqueueNDRangeKernel(queue, copi_k, 1, NULL,
	&global_item_size, &local_item_size, 1, &event2, &event3);
    if ((int) ts%(int)dft*500==0)
    {
        clEnqueueReadBuffer(queue, z,
            CL_TRUE, 0, nn * sizeof(float), Z, 1, &event3, NULL);
                 double dU;
                double mU;
                            mU=9999;
            dU=9999;
        for (int is=kloc;is<kloc+10;is++)
        {

            if (is>0&&is<nn-1)
            {
                
                dU=(Z[is-1]+Z[is+1]-2*Z[is])/(dx*dx);
                if (dU<0)
                {
                    //dU=-dU;
                }
                        
                if (dU<mU)
                {
                    mU=dU;
                    kloc=is;
                }
            }
        }
    }
     }
        clEnqueueReadBuffer(queue, z,
            CL_TRUE, 0, nn * sizeof(float), Z, 1, &event3, NULL);
    clFlush(queue);
    clFinish(queue);
    clReleaseKernel(SP_k);
    clReleaseKernel(SP2_k);
    clReleaseKernel(copi_k);
    clReleaseMemObject(z);
    clReleaseMemObject(zi);
    clReleaseMemObject(pc);
    clReleaseMemObject(a);
    clReleaseMemObject(z2);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    
    


    plhs[0] = mxCreateNumericMatrix(1, nn,mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(1, 1,mxDOUBLE_CLASS, mxREAL);

    double* data = (double *) mxGetData(plhs[0]);
    double* data2= (double*) mxGetData(plhs[1]);
    
    for (i =0;i<nn;i++)
    {
        data[i] = (cl_double) Z[i];
        
        
    }
    printf("%u",kloc);
                    printf("\n");
    *data2=(double)kloc;
    
};
