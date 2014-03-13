//OpenCl Source Code
#define _NUMSTATES 8
#define _NUMTRANS 16
#define _K 3072
#define _N 6144
#define _NUMSUB 192
#define _LSUB 16
#define _GUARDINTRVAL 0	

/*LTE recursive code*/

float addLogs(float A, float B)
{
		float maxi = fmax(A,B);
		float corr = native_log(1.0 + native_exp(-fabs(A-B)));
		return maxi + corr; 
}
/******************************************************************************/ 
__kernel void creategamma(__global float *llrIN, 
											 	  __global float *gamma)
{
	unsigned int _outputs[_NUMTRANS] = {0, 0, 1, 1, 1, 1, 0, 0, 3, 3, 2, 2, 2, 2, 3, 3};
	float sgn[2] = {1, -1};
  float llr[2];
	float logP[4];
	unsigned int t = get_local_id(0);
	unsigned int k = get_group_id(0);

  llr[0] = llrIN[(k<<1)];
	llr[1] = llrIN[(k<<1) + 1];
		
	logP[0] = native_log((1/(1+native_exp(sgn[_outputs[t]>>1] * llr[0]))));
	logP[1] = native_log((1/(1+native_exp(sgn[_outputs[t]&1] * llr[1]))));
	logP[2] = native_log((1/(1+native_exp(sgn[_outputs[t+8]>>1] * llr[0]))));
	logP[3] = native_log((1/(1+native_exp(sgn[_outputs[t+8]&1] * llr[1]))));

	gamma[k * _NUMTRANS + t] = logP[0] + logP[1];
	gamma[k * _NUMTRANS + t + 8] = logP[2] + logP[3];
}
/******************************************************************************/
__kernel void createMatrices(__global float *alpha,
													   __global float *beta,		
													   __global float *gamma)
{
  int _outputs[_NUMTRANS] = {0, 0, 1, 1, 1, 1, 0, 0, 3, 3, 2, 2, 2, 2, 3, 3};
  int _nextStates[_NUMTRANS] = {0, 4, 5, 1, 2, 6, 7, 3, 4, 0, 1, 5, 6, 2, 3, 7};
  int _prevStates[_NUMTRANS] = {0, 3, 4, 7, 1, 2, 5, 6, 1, 2, 5, 6, 0, 3, 4, 7};
	int s = get_local_id(0);
	int mID = get_group_id(0);
		
	if( mID == 1)
	{	
		int	prevStateA = 	_prevStates[s];
		int prevStateB = 	_prevStates[s + _NUMSTATES];
		int transitionA = _prevStates[s];
		int transitionB =	_prevStates[s] + _NUMSTATES;
		for(int k = 0; k < _K-1; k++ )
		{
		
			alpha[(k+1)*_NUMSTATES+s] = addLogs(alpha[prevStateA + k*_NUMSTATES] + gamma[transitionA + k*_NUMTRANS], 
										        	      		  alpha[prevStateB + k*_NUMSTATES] + gamma[transitionB + k*_NUMTRANS]);		
		}	
	}
	else
	{
		int nextStateA = _nextStates[s];
		int nextStateB = _nextStates[s + _NUMSTATES];
		int transitionBetaA = s;
		int transitionBetaB =	s + _NUMSTATES;
		for (int k = _K-1; k > 0; k--)
		{
			beta[k*_NUMSTATES+s] = addLogs(beta[nextStateA + (k+1)*_NUMSTATES] + gamma[transitionBetaA + k*_NUMTRANS], 
										              	 beta[nextStateB + (k+1)*_NUMSTATES] + gamma[transitionBetaB + k*_NUMTRANS]);
		}
	}
}


__kernel void matricesSubDecoder(__global float *alpha,
																 __global float *beta,
																 __global float *gamma)
{
  int _outputs[16] = {0, 0, 1, 1, 1, 1, 0, 0, 3, 3, 2, 2, 2, 2, 3, 3};
  int _nextStates[16] = {0, 4, 5, 1, 2, 6, 7, 3, 4, 0, 1, 5, 6, 2, 3, 7};
  int _prevStates[16] = {0, 3, 4, 7, 1, 2, 5, 6, 1, 2, 5, 6, 0, 3, 4, 7};

  float logStates = -native_log(8.0);
  __local float ATmp[2*_NUMSTATES];
  __local float BTmp[2*_NUMSTATES];

  int s = get_local_id(0);
  int subDec = get_group_id(0);

  int decoderStart = subDec*_LSUB;

  int	prevStateA = 	_prevStates[s];
	int prevStateB = 	_prevStates[s + _NUMSTATES];
	int transitionA = _prevStates[s];
  int transitionB =	_prevStates[s] + _NUMSTATES;

  int nextStateA = _nextStates[s];
	int nextStateB = _nextStates[s + _NUMSTATES];
	int transitionBetaA = s;
	int transitionBetaB =	s + _NUMSTATES;

	ATmp[s] = logStates;
	BTmp[s] = logStates;
  
  if(subDec != 0) 
  {

  	for(int k = _GUARDINTRVAL; k > 0; --k)
  	{
  		ATmp[_NUMSTATES + s] = addLogs(ATmp[prevStateA] + gamma[transitionA + (decoderStart-k)*_NUMTRANS], 
										                 ATmp[prevStateB] + gamma[transitionB + (decoderStart-k)*_NUMTRANS]); 

  		ATmp[s] = ATmp[_NUMSTATES + s];
  	}	
  	alpha[decoderStart*_NUMSTATES + s] = ATmp[s];
  }
  if(subDec != _NUMSUB-1)
  {
  	for(int k = _GUARDINTRVAL; k > 0; --k)
  	{
  		BTmp[_NUMSTATES + s] = addLogs(BTmp[nextStateA] + gamma[transitionBetaA + (decoderStart+_LSUB+k-1)*_NUMTRANS], 
								 								     BTmp[nextStateB] + gamma[transitionBetaB + (decoderStart+_LSUB+k-1)*_NUMTRANS]);
  		BTmp[s] = BTmp[_NUMSTATES + s];
  	}
  	beta[(decoderStart+_NUMSUB)*_NUMSTATES + s] = BTmp[s];
  }

 	for(int k = decoderStart; k < decoderStart + _LSUB - 1; k++)
 	{
		int _k = decoderStart + _LSUB - 1 -(k-decoderStart);
		alpha[(k+1)*_NUMSTATES+s] = addLogs(alpha[prevStateA + k*_NUMSTATES] + gamma[transitionA + k*_NUMTRANS],
									        	    		    alpha[prevStateB + k*_NUMSTATES] + gamma[transitionB + k*_NUMTRANS]);
		
		beta[_k*_NUMSTATES+s] = addLogs(beta[nextStateA + (_k+1)*_NUMSTATES] + gamma[transitionBetaA + _k*_NUMTRANS], 
								            				beta[nextStateB + (_k+1)*_NUMSTATES] + gamma[transitionBetaB + _k*_NUMTRANS]);	
		
 	}

}


/******************************************************************************/
__kernel void llrOut(__global float *alpha,
										 __global float *beta,
										 __global float *gamma,
										 __global float *llrOUT)
{
	int _outputs[16] = {0, 0, 1, 1, 1, 1, 0, 0, 3, 3, 2, 2, 2, 2, 3, 3};
  int _nextStates[16] = {0, 4, 5, 1, 2, 6, 7, 3, 4, 0, 1, 5, 6, 2, 3, 7};	
	float sum[4];
	float sumTmp;
	unsigned int k = get_global_id(0);

	sum[0] = -MAXFLOAT;
	sum[1] = -MAXFLOAT;
	sum[2] = -MAXFLOAT;
	sum[3] = -MAXFLOAT;
	
	for(int t = 0; t < _NUMTRANS; t++)
	{
		sumTmp = gamma[_NUMTRANS*k + t] + beta[ _NUMSTATES*(k+1) + _nextStates[t]] + alpha[_NUMSTATES*k + (t&(_NUMSTATES-1))];
		sum[_outputs[t]>>1] = addLogs(sum[_outputs[t]>>1], sumTmp);
		sum[((_outputs[t]&1)) + 2] = addLogs(sum[((_outputs[t]&1)) + 2], sumTmp);
	}
	
	llrOUT[k]   = sum[1] - sum[0];
	llrOUT[k + _K] = sum[3] - sum[2];
}

__kernel void dataOut(__global float *alpha,
										 __global float *beta,
										 __global float *gamma,
										 __global float *dataOUT)
{
	int _outputs[16] = {0, 0, 1, 1, 1, 1, 0, 0, 3, 3, 2, 2, 2, 2, 3, 3};
  int _nextStates[16] = {0, 4, 5, 1, 2, 6, 7, 3, 4, 0, 1, 5, 6, 2, 3, 7};
	unsigned int k = get_global_id(0);
	float sumA0 = -MAXFLOAT;
	float sumA1 = -MAXFLOAT;

	for(int t = 0; t < _NUMTRANS; t++)
	{
		float gammaTmp = gamma[_NUMTRANS*k + t];
		float betaTmp  = beta[ _NUMSTATES*(k+1) + _nextStates[t]];
		float alphaTmp = alpha[_NUMSTATES*k + (t%_NUMSTATES)];
		int outputTmp  = _outputs[t];		
		
		if(t<_NUMSTATES)
		{
			sumA0 = addLogs(sumA0, gammaTmp+alphaTmp+betaTmp);
		}		
		else
		{
			sumA1 = addLogs(sumA1, gammaTmp+alphaTmp+betaTmp);
		}
	}
	
	(sumA1<sumA0) ? (dataOUT[k]=0) : (dataOUT[k]=1);
}
