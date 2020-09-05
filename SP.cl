


kernel void SP(global float* pc,global float *Z,global float* A,global float* Zi,global float* z2)
{
if (1)
{
size_t i=get_global_id(0);


    if (i<pc[6]-1)
    {
        Z[i] = Zi[i]-(pc[1])/pow(pc[0],pc[3])*(pc[4])*pow(A[i],pc[2])*pow(fabs(Zi[i]-Zi[i+1]),pc[3]);
        z2[i]=Z[i];
        
    }

}
}

kernel void SP2( global float*pc,global float *Z,global float* A,global float* Zi,global float*z)
{
    size_t i=get_global_id(0);


if(i<(int)pc[6]-1) 
{

if(i>0 )

    {
        Z[i] = (z[i]+Zi[i])/2-pc[1]/(2*pow(pc[0],pc[3]))*pc[4]*pow(A[i+1],pc[2])*pow(fabs(z[i-1]-z[i]),pc[3]);
    

    if( Z[i]>Zi[i-1])
    {
        Z[i]=Zi[i-1];
    }
    }
else
    {
        if (pc[1]<0)
        {      
        Z[0]=Zi[0]-pc[0]/pow(pc[0],pc[3])*pow(A[0],pc[2])*pow(fabs(Zi[0]-Zi[1]),pc[3]);
        }
    } 
    if (Z[i]<Zi[i+1])
    {
        Z[i]=Zi[i+1];
    }
Z[i]+=pc[5]*pc[1];

}
else
{
    if (pc[1]<0)
    {
    
    Z[i]=Zi[i]-pc[1]/pow(pc[0],pc[3])*pc[4]*pow(A[i],pc[2])*pow(Zi[i-1]-Zi[i],pc[3]);
    Z[i]+=pc[5]*pc[1];

    }
}
}

kernel void copi(global float *Z,global float* Zi)
{

size_t i=get_global_id(0);

Zi[i]=Z[i];
}



