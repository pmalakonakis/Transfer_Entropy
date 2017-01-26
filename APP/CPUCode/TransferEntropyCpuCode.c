#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pthread.h>

#include "Maxfiles.h"
#include "MaxSLiCInterface.h"

#define THREAD_COUNT 4

#define NUM_VARS 4 //Num of random variables

int M = 10000;  //time series length
int resol = 100; //num of histogram Bins

float ret[NUM_VARS][NUM_VARS];

void equiwidthhist(float *x,float *y,int size, float *probx, float *probxp1_x, float *probx_y,float *probxp1_x_y);

typedef struct {
	int var1;
	int var2;
	int id;
	float *x;
	float *y;
	max_engine_t *engine;
} arg_t;

void equiwidthhist(float *x,float *y,int size, float *probx, float *probxp1_x, float *probx_y,float *probxp1_x_y){

	float maxx=0,maxy=0,minx,miny;
	int i,l,k,st=0;
	int R = size;

	//calculate min-max
	miny = y[1];
	minx=x[1];
	for (i=0; i<M; i++){
		if (maxx<x[i])
			maxx=x[i];

		if (maxy<y[i])
			maxy=y[i];

		if (minx>x[i])
			minx=x[i];

		if (miny>y[i])
			miny=y[i];
	}

	//histogram
	for (i=0; i<M; i++){

		if (maxx==minx){
			k = 1;
			st=1;
		}
		else{
			k = (int)(R * (x[i]-minx)/(maxx-minx));
			if ((i+1)<M)
				st = (int)(R * (x[i+1]-minx)/(maxx-minx));
		}

		if (maxy==miny)
			l = 1;
		else
			l = (int)(R * (y[i]-miny)/(maxy-miny));

		if (k>=0 && k<R){
			probx[k] +=1.0;
		}

		if (k>=0 && k<R)
			if (st>=0 && st<R){
				probxp1_x[st+R*k] +=1.0;
			}

		if (k>=0 && k<R)
			if (l>=0 && l<R){
				probx_y[k+R*l]+=1.0;
			}

		if (k>=0 && k<R)
			if (l>=0 && l<R)
				if (st>=0 && st<R ){
					probxp1_x_y[st+R*k+R*R*l]+=1.0;
				}
	}


}

float TEsoft(float *x,float *y)
{
	const long int resoln = resol;
	long int resolnr = 0;
	resolnr=((resoln*4)+48-1)/48;
	resolnr=48*resolnr/4;
	const long int size = resolnr;

	long int R = size;

	float *probx= calloc(size,4);
	float *probxp1_x= calloc(size*size,4);
	float *probx_y= calloc(size*size,4);
	float *probxp1_x_y= calloc(size*size*size,4);

	float logs;
	float t_entropy=0.0;
	int i,j,z;

	//pdf estimation equiwidth hist
	equiwidthhist( x, y, size, probx, probxp1_x, probx_y,probxp1_x_y);

	//calculate TE
	for (z=0; z<R; z++)
		for (j=0; j<R; j++)
			for (i=0; i<R; i++){
				logs = log2(((float)(probxp1_x_y[i+R*j+R*R*z]/M)) *(((float)(probx[j]/M))) / (((float)(probx_y[j+R*z]/M)) * ((float)(probxp1_x[i+R*j]/M))));
				if (isfinite(logs))
					t_entropy = t_entropy + ((float)probxp1_x_y[i+R*j+R*R*z]/M) * logs;
			}

	free (probx);
	free (probx_y);
	free (probxp1_x_y);
	free (probxp1_x);

	return t_entropy;
}



void *TE(void *arg)
{

	float *x = ((arg_t *)arg)->x;
	float *y = ((arg_t *)arg)->y;
	max_engine_t *engine = ((arg_t *)arg)->engine;
	int var1 = ((arg_t *)arg)->var1;
	int var2 = ((arg_t *)arg)->var2;

	const long int resoln = resol;
	long int resolnr = 0;
	resolnr=((resoln*4)+48-1)/48;
	resolnr=48*resolnr/4;

	const long int size = resolnr;
	long int  sizeBytes = size * sizeof(float);

	float *s = malloc(size*size*sizeBytes);
	long int R = size;

	float *probx= calloc(size,4);
	float *probx_y= calloc(size*size,4);
	float *probxp1_x_y= calloc(size*size*size,4);
	float *probxp1_x= calloc(size*size,4);

	int i,z;

	//pdf estimation equiwidth hist
	equiwidthhist( x, y, size, probx, probxp1_x, probx_y,probxp1_x_y);

/// run on DFE
	max_file_t *maxfile = TransferEntropy_init();
	max_actions_t *act = max_actions_init(maxfile, "default");

	max_set_param_uint64t(act, "N", size);
	max_set_param_uint64t(act, "Ns", size*size*size/3);
	max_set_param_double(act, "m",(float) M);


	max_queue_input(act, "probxp1_x_y", probxp1_x_y, size* size* sizeBytes/3);

	max_queue_input(act, "probx_y", probx_y,size*sizeBytes/3); // multiple of 16

	for (i=0; i<R/3; i++)
		max_queue_input(act, "probx", probx, sizeBytes);

	for (z=0; z<R/3; z++)
		max_queue_input(act, "probxp1_x", probxp1_x,size* sizeBytes);

	max_queue_output(act, "s", s,size* size * sizeBytes/3);

	max_queue_input(act, "probxp1_x_y2", &probxp1_x_y[R/3*R*R], size* size* sizeBytes/3);

	max_queue_input(act, "probx_y2", &probx_y[R/3*R],size* sizeBytes/3); // multiple of 16

	max_queue_input(act, "probxp1_x_y3", &probxp1_x_y[2*R/3*R*R], size* size* sizeBytes/3);

	max_queue_input(act, "probx_y3", &probx_y[R*2*R/3],size* sizeBytes/3); // multiple of 16

	max_run(engine,act);

		//calculate TE (Sum)
	float teres = 0.0;
	for (z=R-13; z<R; z++){
		teres= teres + s[R*(R-1)+R*((R/3)-1)*R+z];
	}

	free (probx);
	free (probx_y);
	free (probxp1_x_y);
	free (probxp1_x);
	free (s);

	ret[var1][var2] = teres;
	pthread_exit(NULL);
	return NULL;
}




int main(){

	struct timeval start, stop;
	pthread_t threads[THREAD_COUNT];
	float  a = 11.0;
	int i,j,h;

	float **rvar =(float **)malloc(NUM_VARS * sizeof(float *));
	for (i=0; i<NUM_VARS; i++)
		rvar[i] = (float *)malloc(M * sizeof(float));

	float **tesft =(float **)malloc(NUM_VARS * sizeof(float *));
	for (i=0; i<NUM_VARS; i++)
		tesft[i] = (float *)malloc(NUM_VARS * sizeof(float));

	max_file_t *maxfile = TransferEntropy_init();

	max_engine_t *engine1=max_load(maxfile, "*");
	max_engine_t *engine2=max_load(maxfile, "*");
	max_engine_t *engine3=max_load(maxfile, "*");
	max_engine_t *engine4=max_load(maxfile, "*");

/////////////////////////////////////////////////////////////////////////////////////////
//generate random data as time series
	for (j=0; j<NUM_VARS; j++)
		for (i=0; i<M; i++){
			rvar[j][i]= ((float)rand()/(float)(RAND_MAX/a));
		}

/////////////////////////////////////////////////////////////////////////////////////////
///software call
///////////////////////////////////////////////////////////////////////////////
	gettimeofday(&start, NULL);
	for (j=0; j<NUM_VARS; j++)
		for (h=0; h<NUM_VARS; h++){
			if (j==h)
				continue;
			tesft[j][h]=TEsoft(rvar[j],rvar[h]);
		}
	gettimeofday(&stop, NULL);
	printf("END - Time Software:    %ld μs\n", ((stop.tv_sec * 1000000 + stop.tv_usec)- (start.tv_sec * 1000000 + start.tv_usec))/2);
	printf("Done.\n");

////////////////////////////////////////////////////////////////////////////
///hardware call
////////////////////////////////////////////////////////////////////////////
	gettimeofday(&start, NULL);
	arg_t arg[4];
	arg[0].engine=engine1;
	arg[1].engine=engine2;
	arg[2].engine=engine3;
	arg[3].engine=engine4;
	arg[0].id=0;
	arg[1].id=1;
	arg[2].id=2;
	arg[3].id=3;

	int threadsstart=0;
	j=0;
	h=1;
	for (int k=0; k<(NUM_VARS*(NUM_VARS-1)); k+=THREAD_COUNT){
		for (i = 0; i < THREAD_COUNT; i++){
			if (j==h){
				h=h+1;
			}
			if (h>=NUM_VARS){
				j=j+1;
				h=0;
				if (j>=NUM_VARS)
						break;
			}
			if (h>=NUM_VARS)
				break;
			arg[i].x=rvar[j];
			arg[i].y=rvar[h];
			arg[i].var1=j;
			arg[i].var2=h;
			pthread_create(&threads[i], NULL, &TE, (void*)&arg[i]);
			threadsstart++;
			h=h+1;
		}
			for (int i = 0; i < threadsstart; ++i)
				pthread_join(threads[i], NULL);
			threadsstart=0;
	}

	max_unload(engine1);
	max_unload(engine2);
	max_unload(engine3);
	max_unload(engine4);
	gettimeofday(&stop, NULL);
	printf("END - Time DFE:    %ld μs\n", ((stop.tv_sec * 1000000 + stop.tv_usec)- (start.tv_sec * 1000000 + start.tv_usec))/2);
	printf("Done.\n\n");

	printf("TE Results \n");
	printf("Hardware \tSoftware \n");

	for (i=0;i<NUM_VARS;i++)
		for(j=0;j<NUM_VARS;j++)
			printf("%f\t%f\n",ret[i][j],tesft[i][j]);

	 return 0 ;
}
