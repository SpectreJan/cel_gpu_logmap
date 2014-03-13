/*******************************************************************************
**
** Main file for openDecoder.c
** mainly for Testing Reasons
**
** Jan Krämer Dez 2013
*******************************************************************************/
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "openDecoderConfig.h"
#include "openDecoder.h"

int main(int argc, const char *argv[])
{
	if(argc < 2)
	{
		fprintf(stderr, "Usage: openDecoder <number of runs>\n");
		exit(1);
	}
	int runs = atoi(argv[1]);
	
	//float llrin[32]  = {3.24376,	4.82663,	2.30248,	-1.56122,	0.84069,	-3.92230,	4.06308,	3.79653,	3.17760,	-2.39272, 0.94356,	-4.77487,	-0.74740,	-1.87281,	-3.38515,	-3.21233,	-0.77114,	-4.05770,	0.98523,	-0.29075,	1.95949,	1.99887,	1.38530,	-4.663961,	-4.31193,	-1.80400,	0.30864,	1.54445,	-0.92380,	3.19981,	2.18358,	4.68649};

  fprintf(stdout, " _____                   ______                   _            \n");
  fprintf(stdout, "|  _  |                  |  _  \\                 | |           \n");
  fprintf(stdout, "| | | |_ __   ___ _ __   | | | |___  ___ ___   __| | ___ _ __  \n");
  fprintf(stdout, "| | | | '_ \\ / _ \\ '_ \\  | | | / _ \\/ __/ _ \\ / _` |/ _ \\ '__| \n");
  fprintf(stdout, "\\ \\_/ / |_) |  __/ | | | | |/ /  __/ (_| (_) | (_| |  __/ |    \n");
  fprintf(stdout, " \\___/| .__/ \\___|_| |_| |___/ \\___|\\___\\___/ \\__,_|\\___|_|    \n");
  fprintf(stdout, "      | |                                                      \n");
  fprintf(stdout, "      |_|\n");
  fprintf(stdout, "\n\n");
  printf("OpenDecoder Configuration:\n");
  printf("->Datapacket: %d bit\n->Coded Packet: %d bit\n->Coderate: 2\n->Subdecoder: %d bit\n->Guardinterval: %d bit\n", 
					_K, _N, _LSUB, _GUARDINTTERVAL);
	
	openDecoder(runs);

	fprintf(stdout, "OpenDecoder finished\nHave a nice day and much fish!\n");

	return 0;
}
