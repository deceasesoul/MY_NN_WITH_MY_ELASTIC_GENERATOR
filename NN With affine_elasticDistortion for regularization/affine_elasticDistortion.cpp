#include "mex.h"
#include<math.h>
#include<algorithm>
#define max(a,b) ( (a) > (b) ? (a): (b))
#define min(a,b) ( (a) < (b) ? (a): (b))
/*
 * This is a MEX-file for MATLAB.
 * Copyright (c) 1984-1998 The MathWorks, Inc.
 */
inline double getPixel(double pic[],double indexrow, double indexcol, int picDim)
{    
        double i=max(0.0001,indexrow);
        i=min(picDim-1.0001,i);
        double j=max(0.0001,indexcol);
        j=min(picDim-1.0001,j);
        int il=(int)floor(i);        
        int ih=il+1;
        double irate=i-(double)il;
        
        int jl=(int)floor(j);
        int jh=jl+1;
        double jrate=j-(double)jl;
        
        double picll=pic[il+jl*picDim];
        double piclh=pic[il+jh*picDim];
        double pichl=pic[ih+jl*picDim];
        double pichh=pic[ih+jh*picDim];
        
        double apicx1=(1-jrate)*picll+jrate*piclh;
        double apicx2=(1-jrate)*pichl+jrate*pichh;
        
        double pixel=(1-irate)*apicx1+irate*apicx2;
        return pixel;
}

void affine_elasticDistortion(double pic[], double model[], double imagesize, double scale, double dpic[])
{
    int imsize=(int)imagesize;
    int index=0;
    int i;
    int j;
    for(i=0; i<imsize; i++) //col
    {
        for(j=0; j<imsize; j++)  //row
        {
            index=imsize*i+j;
            dpic[index]=getPixel(pic,j+scale*model[2*index],i+scale*model[2*index+1],imagesize);
        }
    }    
    //y[0] = 2.0*max(x[0],1);
}
void distortMat(double picmat[],double imagesize,double model[],double scale, double dpic[],double cols[], int colSize)
{
    double* p=NULL;
    double* q=dpic;
    int picLen=(int)imagesize*imagesize;
    for(int i=0; i<colSize; i++)
    {
        p=picmat+ ((int)cols[i]-1)*picLen;           
        affine_elasticDistortion(p,model,imagesize,scale,q);
        q+=picLen;
    }
}


void mexFunction( int nlhs, mxArray *plhs[],  int nrhs, const mxArray *prhs[] )
{
    /*sfsefaesdfaef
     *
     *
     */
    double *pic,*grad,*imagesize,*scale,*dpic,*cols;
    int mrows,ncols,colSize;
    /* Check for proper number of arguments. */
    if(nrhs!=5)
    {
        mexErrMsgTxt("Four input required.");
    }
    else if(nlhs>1) {
        mexErrMsgTxt("Too many output arguments");
    }
    
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    colSize=mxGetN(prhs[4]);
//     printf("%d\n",colSize);
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Input must be a noncomplex scalar double.");
    }
    plhs[0] = mxCreateDoubleMatrix(mrows,colSize, mxREAL);
    pic = mxGetPr(prhs[0]);
    grad=mxGetPr(prhs[1]);
    imagesize=mxGetPr(prhs[2]);
    scale=mxGetPr(prhs[3]);
    cols=mxGetPr(prhs[4]);
    dpic = mxGetPr(plhs[0]);
    distortMat(pic,*imagesize,grad,*scale,dpic,cols,colSize);
}