// reader.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <unistd.h>

void NCHW2NHWC(
  float * samples_ptr, 
  __uint8_t * samples_bytes, 
  int n, 
  int h, 
  int w, 
  int c
){
	int i, j, k, l;
	int t_strides[3], s_strides[3];
	t_strides[0] = h*w*c;
	t_strides[1] = w*c;
	t_strides[2] = c;
	
	s_strides[0] = h*w*c;
	s_strides[1] = h*w;
	s_strides[2] = w;
	
	for(i=0; i<n; ++i){
		for(j=0; j<h; ++j){
			for(k=0; k<w; ++k){
				for(l=0; l<c; ++l){
					samples_ptr[i*t_strides[0]+j*t_strides[1]+k*t_strides[2]+l] = samples_bytes[i*s_strides[0]+l*s_strides[1]+j*s_strides[2]+k]/255.0;
				}
			}
		}
	}
}
