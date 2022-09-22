#include "FastZ.h"

using namespace std; 

#define gapOpen -400 // GapOpen penality
#define gapExtend -30 // GapExtent penality
#define NegInf (-1932735283) // Smallest score possible, as defined in LASTZ.

/**
 * 
 * @param  {char} s : Takes in a character and converts to integer!
 * @return {int}    : 
 */
int alphaToInteger(char s){

	int temp;
	temp = 0;

	switch (s)
   {
	   case 'A': temp = 0;
			   break;
	   case 'C': temp = 1;
				break;
	   case 'G': temp = 2;
			   break;
	   case 'T': temp = 3;
				break;
	   case 'a': temp = 0;
				break;
	   case 'c': temp = 1;
				break;
	   case 'g': temp = 2;
			   break;
	   case 't': temp = 3;
			   break;
	   case ' ': temp = 4;
				break;
	   default: // Default character is for masked repeats which show  up as Ns in the fasta file.
	   			temp = 4;
				break;
   }

   return temp;

}


/**
 * @def: This function takes in two sequences and converts them into integer sequences(stored as char).
 * @param  {std::string} seq1 : input seq1
 * @param  {unsigned} str1len : seq1 length
 * @param  {std::string} seq2 : input seq2
 * @param  {unsigned} str2len : seq2 length
 * @param  {char*} charseq1   : output sequence1 int(stored as char)
 * @param  {char*} charseq2   : output sequence2 int(stored as char)
 */
 void seqToInteger(const std::string& seq1, unsigned str1len, const std::string& seq2, unsigned str2len, char* charseq1, char* charseq2){

	if (str1len < str2len){

		for(int i=0;i<str1len;i++){
			charseq1[i] = (char)(alphaToInteger(seq1[i]));
			charseq2[i] = (char)(alphaToInteger(seq2[i]));
		}

		for(int i=str1len;i<str2len;i++){
			charseq2[i] = (char)(alphaToInteger(seq2[i]));
		}

	}else{

		for(int i=0;i<str2len;i++){
			charseq1[i] = (char)(alphaToInteger(seq1[i]));
			charseq2[i] = (char)(alphaToInteger(seq2[i]));
		}

		for(int i=str2len;i<str1len;i++){
			charseq1[i] = (char)(alphaToInteger(seq1[i]));
		}


	}

}

 /**
  * @param  {int*} gpuScores     	:  Holds the final scores
  * @param  {char*} gpuString1s     :  SEQUENCE1
  * @param  {char*} gpuString2s     :  SEQUENCE2
  * @param  {int} string1 length    :  Anchor points along sequence 1
  * @param  {int} string2 length    :  Anchor points along sequence 2
  * @param  {int*} gpuAnchors1s     :  Anchor points along sequence 1
  * @param  {int*} gpuAnchors2s     :  Anchor points along sequence 2
  * @param  {int*} backup           :  Array to copy overlaps
  * @param  {int} offset            :  Offset within the seeds
  * @param  {int} stream            :  Current stream
  * @param  {int} problemsPerLaunch :  Number of seeds per kernel launch.
  * @param  {int*} gpuTerminations  :  Array to store the final termination points.
  * UNUSED VARS
  * @param  {char*} gpuSmallTB      :  Traceback Memory
  * @param  {char*} gpuTBseq1       :  Memory to store final alignments, for seq1
  * @param  {char*} gpuTBseq2       :  Memory to store final alignments, for seq2
  * @param  {int*} gpuGlobalMem     :  Only necessary for staircase when all three matrices are written to global memory.
  */

  __global__ void CudaFastZTB(int *gpuScores, char *gpuString1s, char *gpuString2s, unsigned str1len, unsigned str2len, int *gpuAnchors1s, int *gpuAnchors2s, int *backup, int offset, int stream, int problemsPerLaunch, int *gpuTerminations, char *gpuSmallTB, char *gpuTBseq1, char *gpuTBseq2, int *gpuGlobalMem, int rev, unsigned maxBench){

    int warpID = (int)((threadIdx.x)/WARPSIZE); // Warp ID within the block.
    char idx = threadIdx.x & 0x1f; // Intra warp thread ID, always ranges from 0 to WARPSIZE-1,
	                     
	int subprobno = stream*INSPROBPERSTREAM+blockIdx.x*INSPROBLEMSPERBLOCK+warpID; // Subproblem number
	int probno = INSPPROBSPEROFFSET*offset+subprobno;                              // Global problem number

	// Fixed scoring matrix
	char gapPenalities[5][5] = {{91,-114,-31,-123,-100},{-114,100,-125,-31,-100},{-31,-125,100,-114,-100},{-123,-31,-114,91,-100},{-100,-100,-100,-100,-100}} ; // HOXD70 substitution scores from LASTZ.

    // Holds the maximum scores seen by each thread and its location.
	// Shared memory as this is shared between threads of the warp.
    __shared__ int maxScore[INSPROBLEMSPERBLOCK][32];
	__shared__ int maxIndex[INSPROBLEMSPERBLOCK][32];

	
    // MEM to hold boundary values during sweeps.
	int* dynamicS1 = &backup[static_cast<unsigned long long>(subprobno)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(MAXPROBSIZE)];
	int* dynamicS2 = &backup[static_cast<unsigned long long>(subprobno)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(MAXPROBSIZE)+static_cast<unsigned long long>(MAXPROBSIZE)];
	int* dynamicI1 = &backup[static_cast<unsigned long long>(subprobno)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(MAXPROBSIZE)+static_cast<unsigned long long>(2)*static_cast<unsigned long long>(MAXPROBSIZE)];
	int* dynamicI2 = &backup[static_cast<unsigned long long>(subprobno)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(MAXPROBSIZE)+static_cast<unsigned long long>(3)*static_cast<unsigned long long>(MAXPROBSIZE)];
    
    // Pointers for boundary values during sweeps.
    int *backupOverlap1 = dynamicS1;
    int *backupOverlap2 = dynamicS2;

    int *backupOverlapI1 = dynamicI1;
    int *backupOverlapI2 = dynamicI2;

    int *tempPtrbackup; // Temporary pointer variable to circular rotate memories, between sweeps.
    
    int i,j; // Holds the translated address.
    int It, Dt, St;
    int delta;

	// These values are shared beween the threads of the warp using warp sync primitives.
    int D_top;
    int I_left;
    int S_top;
    int S_left;
    int S_diag;

    int temp1;
    int temp2;
    int temp3;

    int Sprev;
    int Sprevprev;

    int Iprev;
	int Dprev;
	
    int localMax = 0;           // localMax is the maximum seen by all threads above it.
	int threadMax = NegInf;     // threadMax is the private Maximum seen by a thread.
	int threadIndx = 0;         // Holds the private maximum's index.
	int globalMax = 0;          // globalMax is the maximum score encountered in the DP calculation.
    int globalMaxIndexI = 0;   // Holds the max score index.
    int globalMaxIndexJ = 0;
	int max_i = 0;
    
    // Variables related to the ydrop left and right spearch space.
    int lastCol = 300;
    int startingColumn = 0;
	int endingColumn = 300;
	if(idx > 0){endingColumn = str2len;}
	int endingCol = 300;
    int rowTerminate=0;
    int nextRowEndingColumn=0;
	int finalTerminate = 0;
	int tempEndColumn;
	int beenHere;

	// This overlapping memory is reused
	// Whole warp works on reinitialzing.
	for(int i=idx;i<MAXPROBSIZE;i+=WARPSIZE){

		backupOverlap1[i] = NegInf;
		backupOverlap2[i] = NegInf;
		backupOverlapI1[i] = NegInf;
		backupOverlapI2[i] = NegInf;

	}
    __syncwarp();

    int anchor1; 
	int anchor2; 

	lastCol = 0;
	rowTerminate = 0;
    beenHere = 0; // Only reset been here at this point!
	
	// If you launched more warps than you intended to benchmark.
	// Do not do anything
	if(probno >= maxBench){
		goto ENDSW;
	}

	// Read the anchor points from memory.
	anchor1 = gpuAnchors1s[probno];
	anchor2 = gpuAnchors2s[probno];

	// if this is left extension convert
	// anchors to correspond to reverse strings.
	if(rev == 1){
		anchor1 = str1len - anchor1 - 2;
		anchor2 = str2len - anchor2 - 2;
	}
	
	// Synchronize the localmax.
	localMax = __shfl_sync(0xFFFFFFFF,localMax,31,32);

	// This one loop has been separated to perform traceback for 16x16
	for(int x = 0; x < (WARPSIZE + MAXPROBSIZE) ; x++)	// Loop through all the diagonals.
	{ 
		// Translate from diagonalized coordinates.
		i = (idx); 
		j = x - i;

		if(0xFFFFFFFF == __ballot_sync(0xFFFFFFFF,rowTerminate)){ // break if the row's score does not meet the threshold.
			break;
		}

		tempEndColumn = __shfl_up_sync(0xFFFFFFFF,nextRowEndingColumn,1,32); // Each thread gets the ending column from the thread before it.
		
		localMax = max(localMax,__shfl_up_sync(0xFFFFFFFF,localMax,1,32)); 	// An approximation for a propogating globalMax, ideally the localMax should trickle down 

		// The previous row terminated. Update current endingColumn to the nextRowEndingColumn now!
		if(__shfl_up_sync(0xFFFFFFFF,rowTerminate,1,32) && beenHere == 0 ) 
		{
			beenHere = 1;
			endingColumn = tempEndColumn;
			if(endingColumn < j)	// Already past the column where termination was expected by previous column.
				{
				if(j > lastCol+1){ 	// Terminate current row only, if the previous column did not show any improvement.
					rowTerminate = 1;
					nextRowEndingColumn = lastCol+1;
				}
			}
		}

		// If j is the terminating column
		if( j == endingColumn ){
			if(endingColumn > lastCol+1){ 	// Terminate current row only, if the previous column did not show any improvement.
				rowTerminate = 1;			
				nextRowEndingColumn = lastCol+1;
			}
			else{
				nextRowEndingColumn = endingColumn;
			}
		}


		// ALL threads in the warp are required to participate in the WARP primitive
		// Hence, must be outside a conditional statement.
		temp1 = __shfl_up_sync(0xffffffff, Dprev, 1, 32);
		temp2 = __shfl_up_sync(0xffffffff, Sprev, 1, 32);  
		temp3 = __shfl_up_sync(0xffffffff, Sprevprev, 1, 32);

		// Only enter if, index i and j are valid.	
		if( i >= 0 && ((anchor1 + i + 1) <= ( str1len )) && j >= 0 && ((anchor2 +j + 1) <= ( str2len )) ) 
		{
			
			delta = gapPenalities[static_cast<int>(gpuString2s[anchor2 +j + 1])][static_cast<int>(gpuString1s[anchor1 +i + 1])];
	
			SWCODE // Defined in FastZ.h

			// Reccurance relations
			It = max(I_left+gapExtend,S_left+gapOpen+gapExtend);
			Dt = max(D_top+gapExtend,S_top+gapOpen+gapExtend);
			St = max(It,max(Dt,S_diag+delta));
			
			if(j < endingColumn){ 			//If you are below the column threshold.
				if(St < localMax - 9400)	// If you already hit the Ydrop threshold.
				{
					if(j == startingColumn)	// Ignore the condition till, you have encountered a better score than the starting column.
					{
						startingColumn++;
					}
					else	// Else insert low scores to disable future exploration of a cell.
					{
						It = NegInf;
						St = NegInf;
						Dt = NegInf;
					}
				}
				else
				{
					lastCol = j; // Record column as the last column known, which is not below the ydrop threshold.
				}
			}
			else if((j>=endingColumn) && (rowTerminate != 1) ){ // Hit the column threshold but terminate has not been triggered.
				
				if(It >= localMax - 9400)						// If the score is above the threshold, Insert gaps till it falls below the score.
				{
					nextRowEndingColumn++;
				}
				else{											// Else terminate.
					rowTerminate = 1;
					nextRowEndingColumn = j+1;
				}

			}
			else{ // Else insert low scores to disable future exploration of a cell.
				It = NegInf;
				St = NegInf;
				Dt = NegInf;
			}
			
			if(St >= localMax){		// Update the trickled down localMax if current score is higher
				localMax = St;
		    }

            if(St >= threadMax){	// Update the threadprivate Maximum score and store it in shared memory.
                threadMax = St;
                threadIndx = (((i << 16) & 0xFFFF0000 ) | j);

                maxScore[warpID][idx] = St;
                maxIndex[warpID][idx] = (((i << 16) & 0xFFFF0000 ) | j); // Upper 16 bits have the index i and lower 16 bits have the index j.
                
            }
            
			Sprevprev = Sprev; // Score from two iterations ago.
			Sprev = St;		   // Score previous iteration.
			Iprev = It;		   // Insertion matrix score from previous iteration. 
			Dprev = Dt;		   // Deletion matrix score from previous iteration. 

			// Only thread 31 writes boundary values which are read by thread 0.
			if(idx == 31){
				backupOverlap2[j] = St;
				backupOverlapI2[j] = Dt;
			}

		}

	}//end of threads going across diagonals. Innermost loop.
	
	tempPtrbackup = backupOverlap2;
	backupOverlap2 = backupOverlap1;
	backupOverlap1 = tempPtrbackup;

	tempPtrbackup = backupOverlapI2;
	backupOverlapI2 = backupOverlapI1;
	backupOverlapI1 = tempPtrbackup;

	__syncwarp();

    for(int e=0; e<(int)(SWEEPCOUNT);e++)
    { // Each sweep across all diagonals.
				
        lastCol = 0;
		rowTerminate = 0;
		beenHere = 0;

		// Starting column is the MAXIMUM seen across all threads in a warp.
		startingColumn =  warpMaxaLL(startingColumn);

		//Get the localMax from thread 31 until the thread above gives its localMax.
		localMax = __shfl_sync(0xFFFFFFFF,localMax,31,32);

		endingColumn = __shfl_sync(0xFFFFFFFF,nextRowEndingColumn,31,32);
		endingCol = max(endingCol,endingColumn);
		if(idx > 0){endingColumn = str2len;}

		// If the current thread finds that there is no more feasible region
		// the final terminate flag is set
        if(startingColumn >= endingColumn){
            finalTerminate = 1;
		}

		globalMax =  warpMaxaLL(threadMax); // Get global max from each thread's private MAX.	
		// If any row calls terminate, every thread quits.	
        if(__shfl_sync(0xFFFFFFFF,finalTerminate,0,32) == 1){

			int match = __ballot_sync(0xFFFFFFFF, threadMax == globalMax  );
			int match_index = msbDeBruijn32(match);
			int actualIndex = __shfl_sync(0xFFFFFFFF,threadIndx,match_index,32);

			globalMaxIndexI = ((actualIndex >> 16) & 0x0000FFFF);
			globalMaxIndexJ = ((actualIndex & 0x0000FFFF));

			if(idx == 0){

                for(int i=0;i<WARPSIZE;i++){ 

                    if(globalMax == maxScore[warpID][i]){
                        globalMaxIndexI = ((maxIndex[warpID][i] >> 16) & 0x0000FFFF);
                        globalMaxIndexJ = ((maxIndex[warpID][i]) & 0x0000FFFF);

                        if(globalMaxIndexI != 0 || globalMaxIndexJ != 0){
    
                            if(globalMaxIndexI > max_i ){ // If scores are same, chose the longer alignment by row.
                                max_i = globalMaxIndexI;
                                actualIndex = maxIndex[warpID][i];
                            }
    
                        }
                    }

                }

				gpuTerminations[probno] = actualIndex; // Write the termination point to GMEM.
				if(rev == 1){
					gpuScores[probno] += globalMax;	
				}else{
					gpuScores[probno] = globalMax;
				}
				
			}

			__syncwarp();

            break;
        }


		for(int x = (e*WARPSIZE); x < (e*WARPSIZE  + WARPSIZE + MAXPROBSIZE) ; x++)
        { // Looping across the number of diagonals

            // Translate from diagonalized coordinates.
            i = (e)*WARPSIZE + (idx);
			j = x - i;

			// break if the row's score does not meet the threshold.
			if(0xFFFFFFFF == __ballot_sync(0xFFFFFFFF,rowTerminate)){
				break;
			}
				
			tempEndColumn = __shfl_up_sync(0xFFFFFFFF,nextRowEndingColumn,1,32); // Each thread gets the ending column from the thread before it.
			
			localMax = max(localMax,__shfl_up_sync(0xFFFFFFFF,localMax,1,32)); // An approximation for a propogating globalMax, ideally the localMax should trickle down 
	
			// The previous row terminated. Update current endingColumn to the nextRowEndingColumn now!
			if(__shfl_up_sync(0xFFFFFFFF,rowTerminate,1,32) && beenHere == 0 ) 
			{
				beenHere = 1;
				endingColumn = tempEndColumn;
				if(endingColumn < j)	// Already past the column where termination was expected by previous column.
					{
					if(j > lastCol+1){ 	// Terminate current row only, if the previous column did not show any improvement.
						rowTerminate = 1;
						nextRowEndingColumn = lastCol+1;
					}
				}
			}

			// If j is the terminating column
			if( j == endingColumn ){
				if(endingColumn > lastCol+1){ 	// Terminate current row only, if the previous column did not show any improvement.
					rowTerminate = 1;			
					nextRowEndingColumn = lastCol+1;
				}
				else{
					nextRowEndingColumn = endingColumn;
				}
			}

			// ALL threads in the warp are required to participate in the WARP primitive
			// Hence, must be outside a conditional statement.
			temp1 = __shfl_up_sync(0xffffffff, Dprev, 1, 32);
			temp2 = __shfl_up_sync(0xffffffff, Sprev, 1, 32);  
			temp3 = __shfl_up_sync(0xffffffff, Sprevprev, 1, 32);

			// Only enter if, index i and j are valid.	
			if( i >= 0 && ((anchor1 + i + 1) <= ( str1len )) && j >= 0 && ((anchor2 +j + 1) <= ( str2len )) ) 
			{
						
					 
				delta = gapPenalities[static_cast<int>(gpuString2s[anchor2 +j + 1])][static_cast<int>(gpuString1s[anchor1 +i + 1])];

				SWCODE // Defined in FastZ.h

				// Reccurance relations
				It = max(I_left+gapExtend,S_left+gapOpen+gapExtend);
				Dt = max(D_top+gapExtend,S_top+gapOpen+gapExtend);
				St = max(It,max(Dt,S_diag+delta));
	
				if(j < endingColumn){ 			//If you are below the column threshold.
					if(St < localMax - 9400)	// If you already hit the Ydrop threshold.
					{
						if(j == startingColumn)	// Ignore the condition till, you have encountered a better score than the starting column.
						{
							startingColumn++;
						}
						else	// Else insert low scores to disable future exploration of a cell.
						{
							It = NegInf;
							St = NegInf;
							Dt = NegInf;
						}
					}
					else
					{
						lastCol = j; // Record column as the last column known, which is not below the ydrop threshold.
					}
				}
				else if((j>=endingColumn) && (rowTerminate != 1) ){ // Hit the column threshold but terminate has not been triggered.
					
					if(It >= localMax - 9400)						// If the score is above the threshold, Insert gaps till it falls below the score.
					{
						nextRowEndingColumn++;
					}
					else{											// Else terminate.
						rowTerminate = 1;
						nextRowEndingColumn = j+1;
					}
	
				}
				else{ // Else insert low scores to disable future exploration of a cell.
					It = NegInf;
					St = NegInf;
					Dt = NegInf;
				}
                    
				if(St >= localMax){		// Update the trickled down localMax if current score is higher
					localMax = St;
				}
	
				if(St >= threadMax){	// Update the threadprivate Maximum score and store it in shared memory.
					threadMax = St;
					threadIndx = (((i << 16) & 0xFFFF0000 ) | j);
					maxScore[warpID][idx] = St;
					maxIndex[warpID][idx] = (((i << 16) & 0xFFFF0000 ) | j); // Upper 16 bits have the index i and lower 16 bits have the index j.
				}
				

				Sprevprev = Sprev; // Score from two iterations ago.
				Sprev = St;		   // Score previous iteration.
				Iprev = It;		   // Insertion matrix score from previous iteration. 
				Dprev = Dt;		   // Deletion matrix score from previous iteration. 

				// Only thread 31 writes boundary values which are read by thread 0.
				if(idx == 31){
					backupOverlap2[j] = St;
					backupOverlapI2[j] = Dt;
				}

	
			}

			__syncwarp();
        }//end of threads going across diagonals. Innermost loop.
        
		//////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////// CYCLIC-USE-AND-DISCARD ///////////////////////////////////
    	//////////////////////////////////////////////////////////////////////////////////////
		// Cyclic rotation
        tempPtrbackup = backupOverlap2;
        backupOverlap2 = backupOverlap1;
        backupOverlap1 = tempPtrbackup;

        tempPtrbackup = backupOverlapI2;
        backupOverlapI2 = backupOverlapI1;
        backupOverlapI1 = tempPtrbackup;

	} //end of (sweepCnt/wave)

	// Corner case, when threads do not terminate, but hit the end of the loop.
	if(   (globalMaxIndexI == 0 || globalMaxIndexJ == 0) ){

		globalMax =  warpMaxaLL(threadMax);
		int match = __ballot_sync(0xFFFFFFFF, threadMax == globalMax  );
		int match_index = msbDeBruijn32(match);
		int actualIndex = __shfl_sync(0xFFFFFFFF,threadIndx,match_index,32);

		globalMaxIndexI = ((actualIndex >> 16) & 0x0000FFFF);
		globalMaxIndexJ = ((actualIndex & 0x0000FFFF));
		
		// Only one thread writes the final alignment
        if(idx == 0){

            for(int i=0;i<32;i++){ // This is to iterate over the 32

                if(globalMax == maxScore[warpID][i]){
                    globalMaxIndexI = ((maxIndex[warpID][i] >> 16) & 0x0000FFFF);
                    globalMaxIndexJ = ((maxIndex[warpID][i]) & 0x0000FFFF);
                    if(globalMaxIndexI != 0 || globalMaxIndexJ != 0){

                        if(globalMaxIndexI > max_i ){
                            max_i = globalMaxIndexI;
                            actualIndex = maxIndex[warpID][i];
                        }

                    }
                }

            }

            gpuTerminations[probno] = actualIndex;
			if(rev == 1){
				gpuScores[probno] += globalMax;	
			}else{
				gpuScores[probno] = globalMax;
			}
        }


	}

	ENDSW: // Wait for all threads to reach here!
	__syncwarp();

}


__global__ void FastzExecutor(char *gpuString1s, char *gpuString2s, unsigned str1len, unsigned str2len , int *gpuAnchors1s, int *gpuAnchors2s, int *gpuLengths1, int *gpuLengths2, int *backup, int offset, int stream, int problemsPerLaunch, int *gpuTerminations, int *gpuexecutorProbList, char *gpuTBSpace, int maxProb, int previousBins, char *gpuResults, int *gpuGlobalMem, unsigned rev){

	char idx = threadIdx.x & 0x1f;  // Intra warp thread ID, always ranges from 0 to WARPSIZE-1,
	
	int subprobno = offset*problemsPerLaunch+(blockIdx.x); // Subproblem
	
	int probno = gpuexecutorProbList[previousBins+subprobno]; // Global problem number
	
	char GapPenalities[5][5] = {{91,-114,-31,-123,-100},{-114,100,-125,-31,-100},{-31,-125,100,-114,-100},{-123,-31,-114,91,-100},{-100,-100,-100,-100,-100}} ;

    int anchor1 = gpuAnchors1s[probno];
    int anchor2 = gpuAnchors2s[probno];
	// Get alignment lengths!
    int len1 = ((  gpuLengths1[probno] >> 16) & 0x0000FFFF) + 1;
	int len2 = ((  gpuLengths1[probno]) & 0x0000FFFF) + 1;

	//Get the reverse anchor points.
	if(rev == 1){
		
		anchor1 = str1len - anchor1 - 2;
	 	anchor2 = str2len - anchor2 - 2;
	
	}

	char *tbSpace = &gpuTBSpace[(static_cast<unsigned long long>(maxProb)*static_cast<unsigned long long>(maxProb))*static_cast<unsigned long long>(blockIdx.x)];
	
	int* backupOverlap1 = &backup[static_cast<unsigned long long>(blockIdx.x)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(maxProb)];
	int* backupOverlap2 = &backup[static_cast<unsigned long long>(blockIdx.x)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(maxProb)+static_cast<unsigned long long>(maxProb)];
	int* backupOverlapI1 = &backup[static_cast<unsigned long long>(blockIdx.x)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(maxProb)+static_cast<unsigned long long>(2)*static_cast<unsigned long long>(maxProb)];
	int* backupOverlapI2 = &backup[static_cast<unsigned long long>(blockIdx.x)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(maxProb)+static_cast<unsigned long long>(3)*static_cast<unsigned long long>(maxProb)];
    
	int  sweepCnt = ((float)(len1)/(float)(WARPSIZE)); // Counter keeps number of sweeps required.
    if((float)sweepCnt != ((float)(len1)/(float)(WARPSIZE)))
        sweepCnt += 1;

    int *tempPtrbackup;
    
    int i,j; // Translated addresses!
	int It, Dt, St;
	int delta;

	int D_top;
	int I_left;
	int S_top;
	int S_left;
	int S_diag;

	int temp1;
	int temp2;
	int temp3;
	int tempTb;
	int temptbb;
	unsigned link;

	int Sprev;
	int Sprevprev;

	int Iprev;
	int Dprev;

    
	int inval = 0;


	// Only the first thread does this!
	for(int e=0; e<(int)(sweepCnt);e++)
    { // Each sweep across all diagonals.

		for(int x = (e*WARPSIZE); x < (e*WARPSIZE  + WARPSIZE + len2) ; x++)
        {
            // Looping across the number of diagonals
			// Now thread 0 begins doing useful work!
			// Get original DP matrix coordinates
			i = (e)*WARPSIZE + (idx);
			j = x - i;

			temp1 = __shfl_up_sync(0xffffffff, Dprev, 1, 32);
			temp2 = __shfl_up_sync(0xffffffff, Sprev, 1, 32);  // Left element! S[i-1][j]
			temp3 = __shfl_up_sync(0xffffffff, Sprevprev, 1, 32);

			if(  i <= ( len1 ) && i >= 0 && j >= 0  && j <= len2 )
			{
    
                if( ((anchor2 +j + 1) >= str2len) || ((anchor1 + i + 1) >= str1len) ){	
					inval = 1;				
	                    continue;
                }
                
				delta = GapPenalities[static_cast<int>(gpuString2s[anchor2 +j + 1])][static_cast<int>(gpuString1s[anchor1 +i + 1])];

				SWCODE

				It = max(I_left+gapExtend,S_left+gapOpen+gapExtend);
				
				tempTb = 0;
				link = 0;

				temptbb = 1;
				link = cFromC;
				if(I_left+gapExtend > S_left+gapOpen+gapExtend){
					temptbb = 2;
					link = link | iExtend;
				}

				tempTb = tempTb | (temptbb << 2);
				
				Dt = max(D_top+gapExtend,S_top+gapOpen+gapExtend);

				temptbb = 1;
				if(D_top+gapExtend > S_top+gapOpen+gapExtend){
					temptbb = 3;
					link = cFromC | dExtend;
				}

				tempTb = tempTb | (temptbb << 4);
				
				St = max(It,max(Dt,S_diag+delta));

				temptbb = 1;
				if( Dt > S_diag+delta || It > S_diag+delta){

					if(Dt >= It){
						temptbb = 3;
						link = cFromD | iExtend | dExtend;
					}else{
						temptbb = 2;
						link = cFromI | iExtend | dExtend;
					}
				}

				tempTb = tempTb | temptbb;

				tbSpace[(static_cast<unsigned long long>(i+1)*static_cast<unsigned long long>(len2+1))+static_cast<unsigned long long>(j+1)] = (tempTb);
				
				Sprevprev = Sprev;
				Sprev = St;
				Iprev = It;
				Dprev = Dt;

				if(j==(len2-1) && i==(len1-1)){
					goto dp_finish_00;
                }

				if(idx == 31){
					backupOverlap2[j] = St;
					backupOverlapI2[j] = Dt;		
				}

			} //end of thread check



		}//end of threads going across diagonals. Innermost loop.

		tempPtrbackup = backupOverlap2;
		backupOverlap2 = backupOverlap1;
		backupOverlap1 = tempPtrbackup;

		tempPtrbackup = backupOverlapI2;
		backupOverlapI2 = backupOverlapI1;
		backupOverlapI1 = tempPtrbackup;

		__syncwarp();

	} //end of (sweepCnt/wave)

	dp_finish_00:

	int origin;
	int Op = 1;
	int a;
	int b;

	// These condatin the optiomal alignments of (i,j)!
	a = len1;
	b = len2;

	char *final1; 

	if(rev == 0){
		final1 = &gpuResults[static_cast<unsigned long long>(subprobno)*static_cast<unsigned long long>(2)*static_cast<unsigned long long>(maxProb+maxProb)+static_cast<unsigned long long>(maxProb+maxProb)];
	
	}
	else{
		final1 = &gpuResults[static_cast<unsigned long long>(subprobno)*static_cast<unsigned long long>(2)*static_cast<unsigned long long>(maxProb+maxProb)];
	}

	unsigned finali = 0;
	
	// Only one thread performs the traceback.
	if(idx == 0 && inval == 0){

		if(idx == 0){

			// Row 0 till the end of the columns!
			for(int t = 1; t <= (len2 + 1) ; t++)
				tbSpace[(static_cast<unsigned long long>(0)*static_cast<unsigned long long>(len2+1))+static_cast<unsigned long long>(t)] = 2;
	
			for(int u = 1; u <= (len1 + 1) ; u++)
				tbSpace[(static_cast<unsigned long long>(u)*static_cast<unsigned long long>(len2+1))+static_cast<unsigned long long>(0)] = 3;
	
		}

		origin = tbSpace[(static_cast<unsigned long long>(a)*static_cast<unsigned long long>(len2+1))+static_cast<unsigned long long>(b)] & 0x3;

		while(( a <= (len1) && b <= (len2)) && ((a>=1) && (b>=0)) )
		{


			if(Op == 1){
				if(origin == 1){

					final1[finali] = cFromC;

					finali += 1;
					a-=1;
					b-=1;
					Op = origin;
	


					if((a) >= 0 && (b) >= 0){
						origin = tbSpace[(static_cast<unsigned>(a)*static_cast<unsigned>(len2+1))+static_cast<unsigned>(b)] & 0x3;
					}
	
				}
				else{
	
					Op = origin;
	
					if((a) >= 0 && (b) >= 0){

						origin = (tbSpace[(static_cast<unsigned>(a)*static_cast<unsigned>(len2+1))+static_cast<unsigned>(b)] >> (2*(Op-1))) & 0x3;

					}
	
				}
	
			}
			else if(Op == 2){
	
				final1[finali] = cFromI;
	
				finali += 1;
	
				b -= 1;
				Op = origin;
	
				if((a) >= 0 && (b) >= 0){
					origin = (tbSpace[(static_cast<unsigned>(a)*static_cast<unsigned>(len2+1))+static_cast<unsigned>(b)] >> (2*(Op-1))) & 0x3;	
				}
				
			}
			else{

				final1[finali] = cFromD;
	
				finali += 1;
				a -= 1;
				Op = origin;
	
				if((a) >= 0 && (b) >= 0){

					origin = (tbSpace[(static_cast<unsigned>(a)*static_cast<unsigned>(len2+1))+static_cast<unsigned>(b)] >> (2*(Op-1))) & 0x3;
				}
				
		
			}				
		
		}

		if(a > 1 || b > 0){
			if(a > 1){

				for(int t=a;t>1;t--){
					final1[finali] = 1;
					finali+=1;
				}
			}else{
				for(int t=b;t>0;t--){
					final1[finali] = 1;
					finali+=1;
				}
			}

		}

		if(rev == 0){

			gpuTerminations[probno] = finali;

		}
		else{

			int rtemp = gpuTerminations[probno];

			rtemp = rtemp & 0x0000FFFF;

			gpuTerminations[probno] = ((finali << 16) & 0xFFFF0000) | rtemp;

		}


	}
	

}

/* Function to reverse arr[] from start to end*/
void rvereseArray(char arr[], int start, int end)
{
    while (start < end)
    {
        char temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}

extern "C" int kernel(char * charstr1, char * charstr2, char * revstr1, char * revstr2, int str1len, int str2len, int* anchors1, int* anchors2, int *scores, int *endl, int *endr, unsigned tbonly, unsigned TOTBENCHMARK);

extern "C" int kerneltb(char * charstr1, char * charstr2, char * revstr1, char * revstr2, int str1len, int str2len, int* anchors1, int* anchors2, unsigned maxF, int *scores, int *endl, int *endr, char *tbalign , int *finali, unsigned tbonly, unsigned TOTBENCHMARK);

int kernel(char * charstr1, char * charstr2, char * revstr1, char * revstr2, int str1len, int str2len, int* anchors1, int* anchors2, int *scores, int *endl, int *endr, unsigned tbonly, unsigned TOTBENCHMARK)
{

	clock_t start, end; 

	int* cpuScores = scores;

	////////////////////////////////////////////////////
	///////////// Creating 32 streams! /////////////////
	////////////////////////////////////////////////////

	int n_streams =  32;

	// Allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(	cudaStream_t));
	for (int i = 0 ; i < n_streams ; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamCreate(&(streams[i])));
	}

	///////////////////////////////////////////////////////
	////////////// CUDA Memory Setup //////////////////////
	///////////////////////////////////////////////////////

	// Holds the GPU strings
	char *gpuString1;
	char *gpuString2;
	char *gpuRevString1;
	char *gpuRevString2;
	int *gpuScores;
	int *gpuAnchors1;
	int *gpuAnchors2;
	int *gpuLengths1;
    int *gpuLengths2;
	int *gpuTerminations;
	int *gpuInspectorBackup;

	//  Maybe you could change this when wanting to do reverse strings? So that you could just have a reverse string do everything?
	//String1copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuString1, sizeof(char)* (str1len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuString1, charstr1, sizeof(char)* (str1len) , cudaMemcpyHostToDevice));

	//String2copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuString2, sizeof(char)* (str2len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuString2, charstr2, sizeof(char)* (str2len) , cudaMemcpyHostToDevice));

	//StringRev1copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuRevString1, sizeof(char)* (str1len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuRevString1, revstr1, sizeof(char)* (str1len) , cudaMemcpyHostToDevice));

	//StringRev2copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuRevString2, sizeof(char)* (str2len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuRevString2, revstr2, sizeof(char)* (str2len) , cudaMemcpyHostToDevice));
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////// Memory Setup																 /////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Anchor1
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuAnchors1, sizeof(int)*TOTBENCHMARK ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuAnchors1, anchors1, sizeof(int)*TOTBENCHMARK , cudaMemcpyHostToDevice));

	//Anchor2
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuAnchors2, sizeof(int)*TOTBENCHMARK ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuAnchors2, anchors2, sizeof(int)*TOTBENCHMARK , cudaMemcpyHostToDevice));

	//Len1
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuLengths1, sizeof(int)*TOTBENCHMARK ));
	//CUDA_CHECK_RETURN(cudaMemcpy(gpuLengths1, len1list.data(), sizeof(int)*anchor1_list.size() , cudaMemcpyHostToDevice));

	//Len2
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuLengths2, sizeof(int)*TOTBENCHMARK ));
    //CUDA_CHECK_RETURN(cudaMemcpy(gpuLengths2, len2list.data(), sizeof(int)*anchor1_list.size() , cudaMemcpyHostToDevice));
	
	// Scores
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuScores, sizeof(int)*TOTBENCHMARK ));

	//Termination Memory!
	//CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuTerminations, sizeof(int)*anchor1_list.size() )); // You do not have to copy theese onto the memory!!!

	// MallocManaged Memory is being funny with me! Whats happening?
	CUDA_CHECK_RETURN(cudaMallocManaged(&gpuTerminations, sizeof(int)*TOTBENCHMARK ));
	
    //unsigned long long problemsPerOffset = int(float(static_cast<unsigned long long>(INSPMEM))/float(4*MAXPROBSIZE*sizeof(int)));
    
    unsigned long long problemsPerOffset = static_cast<unsigned long long>(INSPPROBSPEROFFSET);

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuInspectorBackup, static_cast<unsigned long long>(INSPPROBSPEROFFSET)*static_cast<unsigned long long>(4)*static_cast<unsigned long long>(MAXPROBSIZE)*static_cast<unsigned long long>(sizeof(int))));

	#if NOPRINT == 0
	
	printf("[DEBUG] Inspector memory is %llu\n",INSPMEM);
	printf("[DEBUG] Inspector streams are is %llu\n",INSPSTREAMS);
	printf("[DEBUG] Inspector memory required per problem is %llu\n",INSPMEMPERPROB);
	printf("[DEBUG] Inspector problems per stream is %llu\n",INSPROBPERSTREAM);
	printf("[DEBUG] Inspector memproblems per offset is %llu\n",INSPPROBSPEROFFSET);
	printf("[DEBUG] Inspector total offsets are %llu\n",int(ceil((float)TOTBENCHMARK/(float)INSPPROBSPEROFFSET)));
	printf("[DEBUG] Inspector threads per block %llu\n",INSTHREADS);
	printf("[DEBUG] Inspector problems per block %llu\n",INSPROBLEMSPERBLOCK);
	printf("[DEBUG] Inspector number of blocks %llu\n",INSBLOCKS);
	printf("[DEBUG] Inspector RYSIZE is %llu\n",ryMemSize);
	// printf("[DEBUG] Sequence1 size is %llu\n",sizeof(char)* (seq1.length()));
	// printf("[DEBUG] Sequence1 size is %llu\n",sizeof(char)* (seq2.length()));

	#endif

	char* gpuSmallTB;
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuSmallTB, INSPPROBSPEROFFSET*16*16*sizeof(char) ));

	char* gpuTBseq1;
	CUDA_CHECK_RETURN(cudaMallocManaged(&gpuTBseq1, sizeof(char)*(INSPPROBSPEROFFSET)*32 )); 

	char* gpuTBseq2;
	CUDA_CHECK_RETURN(cudaMallocManaged(&gpuTBseq2, sizeof(char)*(INSPPROBSPEROFFSET)*32 )); 
    
    int* gpuGlobalMem;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuGlobalMem, static_cast<unsigned long long>(static_cast<unsigned long long>(TOTBENCHMARK)*static_cast<unsigned long long>(sizeof(int)) )) );

	//////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// LOAD BALANCING USING STREAMS /////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	
	start = clock();

	int remainingProblems = TOTBENCHMARK;

	for(int i=0;i<int(ceil((float)TOTBENCHMARK/static_cast<float>(INSPPROBSPEROFFSET)));i++ )
	{
		if(remainingProblems>=INSPPROBSPEROFFSET){

			for(int  z=0;z<INSPSTREAMS;z++)
			{
				CudaFastZTB<<< INSBLOCKS, INSTHREADS, (INSPROBLEMSPERBLOCK)*(16*16)*sizeof(char), streams[z] >>>(gpuScores, gpuString1, gpuString2, str1len, str2len, gpuAnchors1, gpuAnchors2, gpuInspectorBackup, i, z, problemsPerOffset , gpuTerminations, gpuSmallTB, gpuTBseq1, gpuTBseq2, gpuGlobalMem,0,TOTBENCHMARK);
			
			}
			remainingProblems -= INSPPROBSPEROFFSET;
			
		}
		else{

			for(int z=0; z<static_cast<int>(ceil(static_cast<float>(remainingProblems)/static_cast<float>(INSPROBPERSTREAM))); z++){


				CudaFastZTB<<< INSBLOCKS, INSTHREADS, (INSPROBLEMSPERBLOCK)*(16*16)*sizeof(char), streams[z] >>>(gpuScores, gpuString1, gpuString2, str1len, str2len, gpuAnchors1, gpuAnchors2, gpuInspectorBackup, i, z, problemsPerOffset , gpuTerminations, gpuSmallTB, gpuTBseq1, gpuTBseq2, gpuGlobalMem,0,TOTBENCHMARK);


			}

			remainingProblems -= remainingProblems;

		}
		
	}

	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // This is where you wait for all the inspector kernels to terminate!

	memcpy(endr, gpuTerminations, sizeof(int)*TOTBENCHMARK );

	remainingProblems = TOTBENCHMARK;

	for(int i=0;i<int(ceil((float)TOTBENCHMARK/static_cast<float>(INSPPROBSPEROFFSET)));i++ )
	{
		if(remainingProblems>=INSPPROBSPEROFFSET){

			for(int  z=0;z<INSPSTREAMS;z++)
			{
				CudaFastZTB<<< INSBLOCKS, INSTHREADS, (INSPROBLEMSPERBLOCK)*(16*16)*sizeof(char), streams[z] >>>(gpuScores, gpuRevString1, gpuRevString2, str1len, str2len, gpuAnchors1, gpuAnchors2, gpuInspectorBackup, i, z, problemsPerOffset , gpuTerminations, gpuSmallTB, gpuTBseq1, gpuTBseq2, gpuGlobalMem,1,TOTBENCHMARK);
			
			}
			remainingProblems -= INSPPROBSPEROFFSET;
			
		}
		else{

			for(int z=0; z<static_cast<int>(ceil(static_cast<float>(remainingProblems)/static_cast<float>(INSPROBPERSTREAM))); z++){


				CudaFastZTB<<< INSBLOCKS, INSTHREADS, (INSPROBLEMSPERBLOCK)*(16*16)*sizeof(char), streams[z] >>>(gpuScores, gpuRevString1, gpuRevString2, str1len, str2len, gpuAnchors1, gpuAnchors2, gpuInspectorBackup, i, z, problemsPerOffset , gpuTerminations, gpuSmallTB, gpuTBseq1, gpuTBseq2, gpuGlobalMem,1,TOTBENCHMARK);


			}

			remainingProblems -= remainingProblems;

		}
		
	}

	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // This is where you wait for all the inspector kernels to terminate!

	CUDA_CHECK_RETURN(cudaFree(gpuInspectorBackup)); 
    CUDA_CHECK_RETURN(cudaFree(gpuSmallTB)); 
	CUDA_CHECK_RETURN(cudaFree(gpuGlobalMem)); 
	
	CUDA_CHECK_RETURN(cudaMemcpy(cpuScores, gpuScores, sizeof(int)*TOTBENCHMARK , cudaMemcpyDeviceToHost));

	memcpy(endl, gpuTerminations, sizeof(int)*TOTBENCHMARK );

	end = clock();
	double time_taken;
	time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
	#if NOPRINT == 0
    std::cout << "Time for Inspector was : " << fixed << time_taken << setprecision(5); 
	cout << " sec " << endl; 
	#else
	std::cout << " Inspector TIME :" << fixed << time_taken << "," << setprecision(5); 
	#endif

	// release all stream
	for (int i = 0 ; i < n_streams ; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i]));
	}


	CUDA_CHECK_RETURN(cudaDeviceReset());

	#if NOPRINT == 0
	std::cout << "[DEBUG]: Done." << std::endl;
	#endif

	return 0;
 }



int kerneltb(char * charstr1, char * charstr2, char * revstr1, char * revstr2, int str1len, int str2len, int* anchors1, int* anchors2, unsigned maxF , int *scores, int *endl, int *endr, char *tbalign, int *finali , unsigned tbonly, unsigned TOTBENCHMARK)
{

	clock_t start, end; 

	////////////////////////////////////////////////////
	///////////// CUDA MEMORY SETUP	////////////////////
	////////////////////////////////////////////////////

	//start = clock();

	////////////////////////////////////////////////////
	///////////// Creating 32 streams! /////////////////
	////////////////////////////////////////////////////

	int n_streams =  32;

	// Allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(	cudaStream_t));
	for (int i = 0 ; i < n_streams ; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamCreate(&(streams[i])));
	}

	///////////////////////////////////////////////////////
	////////////// CUDA Memory Setup //////////////////////
	///////////////////////////////////////////////////////

	// Holds the GPU strings
	char *gpuString1;
	char *gpuString2;
	char *gpuRevString1;
	char *gpuRevString2;
	int *gpuScores;
	int *gpuAnchors1;
	int *gpuAnchors2;
	int *gpuLengths1;
    int *gpuLengths2;
	int *gpuTerminations;

	//  Maybe you could change this when wanting to do reverse strings? So that you could just have a reverse string do everything?
	//String1copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuString1, sizeof(char)* (str1len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuString1, charstr1, sizeof(char)* (str1len) , cudaMemcpyHostToDevice));

	//String2copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuString2, sizeof(char)* (str2len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuString2, charstr2, sizeof(char)* (str2len) , cudaMemcpyHostToDevice));

	//StringRev1copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuRevString1, sizeof(char)* (str1len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuRevString1, revstr1, sizeof(char)* (str1len) , cudaMemcpyHostToDevice));

	//StringRev2copy
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuRevString2, sizeof(char)* (str2len) ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuRevString2, revstr2, sizeof(char)* (str2len) , cudaMemcpyHostToDevice));
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////// Memory Setup																 /////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Anchor1
		//Anchor1
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuAnchors1, sizeof(int)*TOTBENCHMARK ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuAnchors1, anchors1, sizeof(int)*TOTBENCHMARK , cudaMemcpyHostToDevice));

	//Anchor2
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuAnchors2, sizeof(int)*TOTBENCHMARK ));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuAnchors2, anchors2, sizeof(int)*TOTBENCHMARK , cudaMemcpyHostToDevice));

	//Len1
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuLengths1, sizeof(int)*TOTBENCHMARK ));
	//CUDA_CHECK_RETURN(cudaMemcpy(gpuLengths1, len1list.data(), sizeof(int)*anchor1_list.size() , cudaMemcpyHostToDevice));

	//Len2
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuLengths2, sizeof(int)*TOTBENCHMARK ));
    //CUDA_CHECK_RETURN(cudaMemcpy(gpuLengths2, len2list.data(), sizeof(int)*anchor1_list.size() , cudaMemcpyHostToDevice));
	
	// Scores
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuScores, sizeof(int)*TOTBENCHMARK ));

	//Termination Memory!
	CUDA_CHECK_RETURN(cudaMallocManaged(&gpuTerminations, sizeof(int)*TOTBENCHMARK ));

	#if NOPRINT == 0
	
	printf("[DEBUG] Inspector memory is %llu\n",INSPMEM);
	printf("[DEBUG] Inspector streams are is %llu\n",INSPSTREAMS);
	printf("[DEBUG] Inspector memory required per problem is %llu\n",INSPMEMPERPROB);
	printf("[DEBUG] Inspector problems per stream is %llu\n",INSPROBPERSTREAM);
	printf("[DEBUG] Inspector memproblems per offset is %llu\n",INSPPROBSPEROFFSET);
	printf("[DEBUG] Inspector total offsets are %llu\n",int(ceil((float)TOTBENCHMARK/(float)INSPPROBSPEROFFSET)));
	printf("[DEBUG] Inspector threads per block %llu\n",INSTHREADS);
	printf("[DEBUG] Inspector problems per block %llu\n",INSPROBLEMSPERBLOCK);
	printf("[DEBUG] Inspector number of blocks %llu\n",INSBLOCKS);
	printf("[DEBUG] Inspector RYSIZE is %llu\n",ryMemSize);
	printf("[DEBUG] Sequence1 size is %llu\n",sizeof(char)* (seq1.length()));
	printf("[DEBUG] Sequence1 size is %llu\n",sizeof(char)* (seq2.length()));

	#endif

	char* gpuTBseq1;
	CUDA_CHECK_RETURN(cudaMallocManaged(&gpuTBseq1, sizeof(char)*(INSPPROBSPEROFFSET)*32 )); 

	char* gpuTBseq2;
	CUDA_CHECK_RETURN(cudaMallocManaged(&gpuTBseq2, sizeof(char)*(INSPPROBSPEROFFSET)*32 )); 
    
	//////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// LOAD BALANCING USING STREAMS /////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// Executor	      ////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////

	ap *orignalProblemArray = new ap[TOTBENCHMARK];  // Holds my original problems.

    //////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// Executor trimming ////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////
	
	for(int t=0;t<TOTBENCHMARK;t++){

		unsigned t1 = ((  endl[t] >> 16) & 0x0000FFFF);
		unsigned t2 = ((  endl[t]) & 0x0000FFFF);

		orignalProblemArray[t].len1l =  t1; // Use the terminations from the GPU.
        orignalProblemArray[t].len2l =  t2; 

		orignalProblemArray[t].anchor1 = anchors1[t];
		orignalProblemArray[t].anchor2 = anchors2[t];

		t1 = ((  endr[t] >> 16) & 0x0000FFFF);
		t2 = ((  endr[t]) & 0x0000FFFF);

		orignalProblemArray[t].len1r =  t1; // Use the terminations from the GPU.
		orignalProblemArray[t].len2r =  t2 ; 
		orignalProblemArray[t].score = scores[t];
        orignalProblemArray[t].len1extra = t1;
		orignalProblemArray[t].len2extra =  t2;
		orignalProblemArray[t].originalProblemNumber = t;

	}

	int bin4iterator = 0;

	ap *bin4 =  new ap[TOTBENCHMARK]; // Bin holds <512 
	ap *binx =  new ap[TOTBENCHMARK]; // Bin holds <512 



	thrust::copy_if(thrust::host, orignalProblemArray, orignalProblemArray + (TOTBENCHMARK), bin4, over_threshold());
	bin4iterator = thrust::count_if(thrust::host, orignalProblemArray, orignalProblemArray + (TOTBENCHMARK), over_threshold());
	
	for(int k=0;k<bin4iterator;k++){
		bin4[k].len1extra = k;
	}

	memcpy(binx,bin4,sizeof(ap)*TOTBENCHMARK );
    thrust::sort(thrust::host, binx, binx + (bin4iterator), myComparator() ) ;

    ap *executorBin =  new ap[TOTBENCHMARK];

	// These are the surviving problems!
	int totalExecutorproblems = bin4iterator;
	
	// So this is the list that I am actually passing.
    int *executorProbList = new int[totalExecutorproblems];
    int *gpuexecutorProbList;

    int totalCounter = 0;

    for(int k=0;k<bin4iterator;k++){
		executorProbList[totalCounter] = bin4[k].originalProblemNumber;
		totalCounter+=1;
	}

	// time_taken = double(end - start) / double(CLOCKS_PER_SEC); 

	
    //////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////// BIN4 /////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////

    cudaThreadSetLimit(cudaLimitMallocHeapSize, 1.0*1024*1024*1024); // This sets the amoount of memory that can be malloced from the gpu

    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuexecutorProbList, sizeof(int)*totalExecutorproblems ));
    CUDA_CHECK_RETURN(cudaMemcpy(gpuexecutorProbList, executorProbList, sizeof(int)*totalExecutorproblems , cudaMemcpyHostToDevice));


    unsigned long long totalMemAvailable = sizeof(char)*8.0*1024*1024*1024;
    int * gpuExecBackup;
    char * resultsBackup;
    char *gpuTBSpace;
    int *globalBackupExecutor;

	
    int currentBinMax = maxF; // This is the maximum value seen by this particular bin!

    unsigned long long currentBinPerProblemTB = static_cast<unsigned long long>(currentBinMax)*static_cast<unsigned long long>(currentBinMax);
    unsigned long long currentBinPerProblemBackup = static_cast<unsigned long long>(4*sizeof(int))*static_cast<unsigned long long>(currentBinMax);
    unsigned long long currentBinPerProblemResults = static_cast<unsigned long long>(2*sizeof(char))*static_cast<unsigned long long>(currentBinMax+currentBinMax);

	unsigned long long currentBinPerProblemGlobalMem = static_cast<unsigned long long>(sizeof(int))*static_cast<unsigned long long>(currentBinMax);

    unsigned long long currentBinMemReq = currentBinPerProblemTB+currentBinPerProblemBackup+currentBinPerProblemResults+currentBinPerProblemGlobalMem;

	unsigned blocksToLaunch = int((double)totalMemAvailable/(double)currentBinMemReq);
	
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuExecBackup, blocksToLaunch*currentBinPerProblemBackup )); // For backup arrays.
	
	// You have to hold the results of the entire batch of surviving alignments!

	CUDA_CHECK_RETURN(cudaMalloc((void **)&resultsBackup, totalExecutorproblems*currentBinPerProblemResults )); // For the results arrays
    CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuTBSpace, static_cast<unsigned long long>(blocksToLaunch)*static_cast<unsigned long long>(currentBinPerProblemTB) ));    // For the traceback
    CUDA_CHECK_RETURN(cudaMalloc((void **)&globalBackupExecutor, static_cast<unsigned long long>(blocksToLaunch)*static_cast<unsigned long long>(currentBinPerProblemGlobalMem) ));    // GlobalMemoryWrites 
	
    int loopie = (int(float(bin4iterator))/((float)blocksToLaunch));

	//return 0;


    #if NOPRINT == 0
    std::cout <<  "The maximuim  number of problems that I can launch are: " << int((double)totalMemAvailable/(double)currentBinMemReq) << " and the memory required per problem : " << currentBinMemReq << std::endl ;
	std::cout << "The number of blocks per launch are " << blocksToLaunch << ": The number of loops are: " << loopie << " and the max problem size I saw was " << currentBinMax << std::endl;
	#endif


	CUDA_CHECK_RETURN(cudaMemcpy(gpuLengths1, endr, sizeof(int)*TOTBENCHMARK , cudaMemcpyHostToDevice));

	start = clock();
	// Right extension

    for(int t = 0; t <  loopie ; t++  ){

        FastzExecutor<<< blocksToLaunch, 32, 0, streams[0] >>>(gpuString1, gpuString2, str1len, str2len, gpuAnchors1, gpuAnchors2, gpuLengths1, gpuLengths2, gpuExecBackup, t, 0, blocksToLaunch , gpuTerminations, gpuexecutorProbList, gpuTBSpace, currentBinMax,0,resultsBackup, globalBackupExecutor, 0);
    }

    if (((bin4iterator % blocksToLaunch) != 0) ){

        FastzExecutor<<< (bin4iterator % blocksToLaunch), 32, 0, streams[0] >>>(gpuString1, gpuString2, str1len, str2len, gpuAnchors1, gpuAnchors2, gpuLengths1, gpuLengths2, gpuExecBackup, loopie, 0, blocksToLaunch , gpuTerminations, gpuexecutorProbList, gpuTBSpace, currentBinMax,0,resultsBackup, globalBackupExecutor, 0);

	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	// The left and right extension can have two different lengths so they can have two different lists!

	CUDA_CHECK_RETURN(cudaMemcpy(gpuLengths1, endl, sizeof(int)*TOTBENCHMARK , cudaMemcpyHostToDevice));

	printf("Total blocks %llu to launch %i and total loops %i\n",currentBinPerProblemResults,blocksToLaunch,loopie);

	// Left extension

    for(int t = 0; t <  loopie ; t++  ){

        FastzExecutor<<< blocksToLaunch, 32, 0, streams[0] >>>(gpuRevString1, gpuRevString2,  str1len, str2len, gpuAnchors1, gpuAnchors2, gpuLengths1, gpuLengths2, gpuExecBackup, t, 0, blocksToLaunch , gpuTerminations, gpuexecutorProbList, gpuTBSpace, currentBinMax,0,resultsBackup, globalBackupExecutor, 1);
    }

    if (((bin4iterator % blocksToLaunch) != 0) ){

        FastzExecutor<<< (bin4iterator % blocksToLaunch), 32, 0, streams[0] >>>(gpuRevString1, gpuRevString2, str1len, str2len, gpuAnchors1, gpuAnchors2, gpuLengths1, gpuLengths2, gpuExecBackup, loopie, 0, blocksToLaunch , gpuTerminations, gpuexecutorProbList, gpuTBSpace, currentBinMax,0,resultsBackup, globalBackupExecutor, 1);

	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	end = clock(); 

	// Now copy the results over to the pointer from the calling function!
	CUDA_CHECK_RETURN(cudaMemcpy(tbalign, resultsBackup, totalExecutorproblems*currentBinPerProblemResults , cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaMemcpy(finali, gpuTerminations,  static_cast<unsigned long long>(TOTBENCHMARK)* static_cast<unsigned long long>(sizeof(int)) , cudaMemcpyDeviceToHost));

	double time_taken;
	time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
	#if NOPRINT == 0
    std::cout << "Time for bin4 was : " << fixed << time_taken << setprecision(5); 
	cout << " sec " << endl; 
	#else
	std::cout <<" KERNEL TB TIME: " << fixed << time_taken << "," << setprecision(5); 
    #endif

	// Releasing unused memory.
    CUDA_CHECK_RETURN(cudaFree(gpuExecBackup));
    CUDA_CHECK_RETURN(cudaFree(resultsBackup)); 
    CUDA_CHECK_RETURN(cudaFree(gpuTBSpace));
    CUDA_CHECK_RETURN(cudaFree(globalBackupExecutor)); 

	// Freeing all memory!
	CUDA_CHECK_RETURN(cudaFree(gpuString1));
	CUDA_CHECK_RETURN(cudaFree(gpuString2));
	CUDA_CHECK_RETURN(cudaFree(gpuAnchors1));
	CUDA_CHECK_RETURN(cudaFree(gpuAnchors2));
	CUDA_CHECK_RETURN(cudaFree(gpuLengths1));
	CUDA_CHECK_RETURN(cudaFree(gpuLengths2));
   
	// Deleting my personal bins!
	delete bin4;
	delete orignalProblemArray;

	// release all stream
	for (int i = 0 ; i < n_streams ; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamDestroy(streams[i]));
	}

	CUDA_CHECK_RETURN(cudaDeviceReset());

	std::cout << "[DEBUG]: Done." << std::endl;

	return 0;
 }






 
 /**
  * Check the return value of the CUDA runtime API call and exit
  * the application if the call has failed.
  */
 static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
 {
	 if (err == cudaSuccess)
		 return;
	 std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	 exit (1);
 }
