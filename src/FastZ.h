#pragma once

#ifndef FASTZ_H
#define FASTZ_H

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <assert.h>
#include <string>
#include <bits/stdc++.h>
#include <iomanip>
#include <cuda_profiler_api.h>
#include <iomanip>
#include <math.h>
#include <unistd.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <bits/stdc++.h> 
#include <fstream>


#define WARPSIZE 32 // Default Warp size.

// Alignments longer than this might be truncated to this.
#define MAXPROBSIZE 30720 // Largest alignment length observed across benchmarks,

/*
 Staircase related setup
*/

#define NOPRINT 1 	// setting this to 1 disables all prints.

#ifndef EXTRAWORK
#define EXTRAWORK 0 // Setting this to 1 enables extra work in the executor!
#endif

#ifndef GLOBALMEM
#define GLOBALMEM 0 // Setting this to 1 enables writing 3 arrays into Global Memory in both the inspector and executor.
#endif

#ifndef SMALLTB
#define SMALLTB 1 	// Setting this to 1 enables performing traceback for small problems and getting rid of them..
#endif



/*
 Inspector kernel related setup
*/
#define INSPSTREAMS 32 // Number of inspector streams, maximum supported for all three devices tested is 32 streams.
#define INSTHREADS 128 // Number of threads in each inspector block.

// Least amount of global memory required for the inspector
#define INSPMEMPERPROBBASE (static_cast<unsigned long long>(MAXPROBSIZE)* \
static_cast<unsigned long long>(4)*static_cast<unsigned long long>(sizeof(int))) 	// This includes four buffers for the Score and Insertion matrices. inclides.

#define INSPMEM static_cast<unsigned long long>( (float(2.0)*1024*1024*1024) ) 		// Global memory for inspector, set this to 1.0 (Ampere , Pascal), 2.0 (Volta)
#define INSPMEMPERPROB (static_cast<unsigned long long>(INSPMEMPERPROBBASE)) 

// Math to calculate, how many problems can be launched 
// for the amount of memory, streams and threads defined.
#define	INSPROBPERSTREAMAPPROX  static_cast<unsigned int>(static_cast<double>(INSPMEM)/static_cast<unsigned long long>(INSPMEMPERPROB))		
#define	INSPROBPERSTREAMTEMP  static_cast<unsigned int>(static_cast<double>(INSPROBPERSTREAMAPPROX)/(INSPSTREAMS))		
#define INSPROBLEMSPERBLOCK int(float(INSTHREADS)/float(WARPSIZE))
#define	INSPROBPERSTREAM  static_cast<unsigned int>(INSPROBLEMSPERBLOCK*static_cast<unsigned int>(static_cast<double>(INSPROBPERSTREAMTEMP)/(INSPROBLEMSPERBLOCK)))		
#define INSBLOCKS int(float(INSPROBPERSTREAM)/(INSPROBLEMSPERBLOCK))
#define INSPPROBSPEROFFSET static_cast<unsigned long long>(static_cast<unsigned long long>(INSPROBPERSTREAM)*static_cast<unsigned long long>(INSPSTREAMS))

#define SWEEPCOUNT (((float)(MAXPROBSIZE)/(float)(WARPSIZE)) + 1) // The maximum number of sweeps that the warp needs to do!


typedef struct AnchorPairs // Strucure to hold information about each seed.
{
	int anchor1;
	int anchor2;
    int len1r;
	int len1l;
    int len1extra;
    int len2r;
	int len2l;
	int len1;
	int len2;
	int score;
    int len2extra;
	int originalProblemNumber; 

} ap;

struct over_threshold
{
	__host__ __device__
	bool operator()(const ap x)
	{
		return (x.score > 3000);
	}

};

// These structures are predicates that are used by the thrust library

	struct myComparator
	{
		__host__ __device__  
		bool operator()(const ap &a, const ap &b) const
		{
			return (a.len1l > b.len1l) || (a.len1r > b.len1r) ;
		}
	};


	struct eqPred
	{
		__host__ __device__  
		bool operator()(const int a, const int b) const
		{
			return a == b;
		}

	};

	struct is_less_than_512
	{
		__host__ __device__
		bool operator()(const ap x)
		{
			return (x.len1 >= 16 && x.len1 <= 512);
		}

	};

	struct is_gt_512_lt_2048
	{
		__host__ __device__
		bool operator()(const ap x)
		{
			return (x.len1 > 512 && x.len1 <=2048 );
		}

	};


	struct is_gt_2048_lt_8192
	{
		__host__ __device__
		bool operator()(const ap x)
		{
			return (x.len1 > 2048 && x.len1 <= 8192 );
		}

	};


	struct is_gt_8192_lt_32768
	{
		__host__ __device__
		bool operator()(const ap x)
		{
			return (x.len1 > 8192 && x.len1 <= 32768 );
		}

	};



// Function prototypes.

int alphaToInteger(char s);

void seqToInteger(const std::string& seq1, unsigned seq1len, const std::string& seq2, unsigned seq2len, char* charseq1, char* charseq2);

void rvereseArray(char arr[], int start, int end);


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

// Inline functions.

// Inline reduction function to find the maximum across a warp
__inline__ __device__
int warpMaxaLL(int val) {
  for (int mask = WARPSIZE/2; mask > 0; mask /= 2)
    val = max(val,__shfl_xor_sync(__activemask(),val,mask,32));
  return val;
}

// Inline reduction function to find the minimum across a warp
__inline__ __device__
int warpMinaLL(int val) {
  for (int mask = WARPSIZE/2; mask > 0; mask /= 2)
    val = min(val,__shfl_xor_sync(__activemask(),val,mask,32));
  return val;
}

// Inline function to find the MSB bit set in an int
// REF: https://bit.ly/34Esjvw
__inline__ __device__
unsigned msbDeBruijn32( unsigned v )
{
    static const int MultiplyDeBruijnBitPosition[32] =
    {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
    };

    v |= v >> 1; 
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    return MultiplyDeBruijnBitPosition[( unsigned )( v * 0x07C4ACDDU ) >> 27];
}

#define cFromC  0		// (c bit is no bits) see note (7)
#define cFromI  1
#define cFromD  2
#define iExtend 4
#define dExtend 8
#define cidBits (cFromC | cFromI | cFromD)

#define op_string(op) ((op==cFromC)? "SUB" : (op==cFromI)? "INS" : (op==cFromD)? "DEL" : "???")

// SW
// Though there is branch divergance
// the branches are short 
#define SWCODE												\
if(j==0){													\
															\
	I_left = NegInf;										\
	S_left = gapOpen + (i+1)*gapExtend;						\
															\
}else{														\
															\
	I_left = Iprev;											\
	S_left = Sprev;											\
}															\
if(idx != 0){												\
	S_diag = (j == 0) ? gapOpen + (i)*gapExtend: temp3;		\
	S_top = temp2;											\
	D_top = temp1;											\
}															\
else{														\
	if(j==0 && i==0){										\
		S_diag = 0;											\
	}														\
	else if( i == 0 ){										\
		S_diag = gapOpen + (j)*gapExtend;					\
	}														\
	else if(j == 0){										\
		S_diag = gapOpen + (i)*gapExtend;					\
	}														\
	else{													\
		S_diag = backupOverlap1[ j - 1];					\
	}														\
	if(i ==0){												\
		S_top = gapOpen + (j+1)*gapExtend;					\
		D_top = NegInf;										\
	}														\
	else{													\
		S_top =    backupOverlap1[j];						\
		D_top =   backupOverlapI1[j];						\
	}														\
}															\


#endif