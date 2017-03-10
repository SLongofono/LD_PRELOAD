#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* CPP Headers */
#include <iostream>
#include <string>
#include <map>

/* Include CUDA header */
#include <cuda.h>
#include <vector_types.h>

/* Control the printing of output */
#define PRINT_MASTER	1

#if PRINT_MASTER
#define PRINT(func, statement)	PRINT_##func(statement)
#else
#define PRINT(func, statement)
#endif

#define PRINT_LAUNCH_CTL(x)
#define PRINT_CONFIG_CTL(x)	
#define PRINT_SETUP_CTL(x)
#define PRINT_SYNCH_CTL(x)
#define PRINT_STREAM_CTL(x)	x
#define PRINT_REGTR_CTL(x)

/* Create typedefs to the target CUDA APIs */
typedef cudaError_t	(*cudaLaunch_t)(const char*);
typedef cudaError_t	(*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
typedef cudaError_t	(*cudaSetupArgument_t)(const void*, size_t, size_t);
typedef cudaError_t	(*cudaDeviceSynchronize_t)(void);
typedef cudaError_t	(*cudaStreamSynchronize_t)(cudaStream_t stream);
typedef void		(*cudaRegisterFunction_t)(void**, const char*, char*,
						  const char*, int, uint3*,
						  uint3*, dim3*, dim3*,
						  int*);

/* Statically declare function pointers for real CUDA APIs */
static cudaLaunch_t 		orig_cudaLaunch = NULL;
static cudaConfigureCall_t 	orig_cudaConfigureCall = NULL;
static cudaSetupArgument_t	orig_cudaSetupArgument = NULL;
static cudaDeviceSynchronize_t	orig_cudaDeviceSynchronize = NULL;
static cudaStreamSynchronize_t	orig_cudaStreamSynchronize = NULL;
static cudaRegisterFunction_t 	orig_cudaRegisterFunction = NULL;

/* Declare a global dictionary for keeping track of CUDA kernels */
std::map<std::string, int> &kernel_cnt () {
	static std::map<std::string, int> _cKernels;
	return _cKernels;
}

std::map<const char*, std::string> &kernel_def () {
	static std::map<const char*, std::string> _dKernels;
	return _dKernels;
}

std::map<std::string, int> &launch_cnt () {
	static std::map<std::string, int> _cLaunches;
	return _cLaunches;
}

/*
 * cudaLaunch
 *
 * @func		: Name of the CUDA kernel to launch
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
extern "C"
cudaError_t cudaLaunch (const char *func)
{
	if (!orig_cudaLaunch) {
		/* Get the pointer to CUDA-defined launch function */
		orig_cudaLaunch = (cudaLaunch_t) dlsym (RTLD_NEXT, "cudaLaunch");
	}

	/* Keep track of the launch counts of kernels */
	if (launch_cnt().find (kernel_def()[func]) == launch_cnt().end()) {
		/* Initialize the hash entry */
		launch_cnt()[kernel_def()[func]] = 1;

		/* Print the function name to stdout */
		PRINT (LAUNCH_CTL, std::cout << "Launching kernel : " << kernel_def()[func] << std::endl);
		PRINT (LAUNCH_CTL, std::cout << "Registrations    : " << kernel_cnt()[kernel_def()[func]] << std::endl);
	} else {
		/* Update the launch count against this kernel */
		launch_cnt()[kernel_def()[func]] += 1;

		/* Print the launch count to stdout */
		PRINT (LAUNCH_CTL, std::cout << "Launching kernel : " << kernel_def()[func] << std::endl);
		PRINT (LAUNCH_CTL, std::cout << "Launches         : " << launch_cnt()[kernel_def()[func]] << std::endl);
	}

	/* Call the cuda laucnh function */
	return orig_cudaLaunch (func);
}

/*
 * cudaConfigureCall
 *
 * @gridDim		: Dimensions of the thread grid
 * @blockDim		: Dimensions of the thread block
 * @sharedMem		: !Amount of shared memory
 * @stream		: ID of the CUDA stream
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
extern "C"
cudaError_t cudaConfigureCall (dim3		gridDim,
			       dim3		blockDim,
			       size_t		sharedMem = 0,
			       cudaStream_t	stream = 0)
{
	if (!orig_cudaConfigureCall) {
		/* Get the pointer to the CUDA-defined configure function */
		orig_cudaConfigureCall = (cudaConfigureCall_t) dlsym (RTLD_NEXT, "cudaConfigureCall");
	}

	/* Print the invocation message */
	PRINT (CONFIG_CTL, std::cout << "Configuring CUDA...\n");

	/* Call the cuda configure function */
	return orig_cudaConfigureCall (gridDim, blockDim, sharedMem, stream);
}

/*
 * cudaSetupArgument
 *
 * @arg			: Argument to pass to CUDA kernel
 * @size		: Size of the argument
 * @offset		: Offset of the argument in the kernel stack
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
extern "C"
cudaError_t cudaSetupArgument (const void	*arg,
			       size_t		size,
			       size_t		offset)
{
	if (!orig_cudaSetupArgument) {
		/* Get the pointer to the CUDA-defined setup function */
		orig_cudaSetupArgument = (cudaSetupArgument_t) dlsym (RTLD_NEXT, "cudaSetupArgument");
	}

	/* Print the invocation message */
	PRINT (SETUP_CTL, std::cout << "Setting up CUDA...\n");

	/* Call the cuda setup function */
	return orig_cudaSetupArgument(arg, size, offset);
}

/*
 * cudaDeviceSynchronize
 *
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
extern "C"
cudaError_t cudaDeviceSynchronize (void)
{
	if (!orig_cudaDeviceSynchronize) {
		/* Get the pointer to the CUDA-defined synchronize function */
		orig_cudaDeviceSynchronize = (cudaDeviceSynchronize_t) dlsym (RTLD_NEXT, "cudaDeviceSynchronize");
	}

	/* Print the invocation message */
	PRINT (SYNCH_CTL, std::cout << "Synchronizing with the device...\n");

	/* Call the original cuda synchronize function */
	return orig_cudaDeviceSynchronize ();
}

/*
 * cudaStreamSynchronize
 *
 * @stream		: Stream identifier
 * $cudaError_t		: Enumerated type specifying the CUDA error
 */
cudaError_t cudaStreamSynchronize (cudaStream_t stream)
{
	if (!orig_cudaStreamSynchronize) {
		/* Get the pointer to the CUDA-defined stream synchronize
		 * function */
		orig_cudaStreamSynchronize = (cudaStreamSynchronize_t) dlsym (RTLD_NEXT, "cudaStreamSynchronize");
	}

	/* Print the invocation message */
	PRINT (STREAM_CTL, std::cout << "Synchronizing with the stream...\n");

	/* Call the original cuda stream synchronization function */
	return orig_cudaStreamSynchronize (stream);
}

/*
 * cudaRegisterFunction
 *
 * @fatCubinHandle	: !PARAM UNKNOWN
 * @hostFun		: !Function to be invoked on host
 * @deviceFun		: !Function to be invoked on device
 *
 * @deviceName		: !Name of the device
 * @threadLimit		: !Maximum number of concurrent threads on the device
 * @tid			: !Identifier for thread
 *
 * @bid			: !Identifier for thread block
 * @bDim		: !Block dimensions
 * @gDim		: !Grid dimensions
 *
 * @wSize		: !Window size
 */
extern "C"
void __cudaRegisterFunction (void 	**fatCubinHandle,
			     const char	*hostFun,
			     char	*deviceFun,
			     const char	*deviceName,
			     int	thread_limit,
			     uint3	*tid,
			     uint3	*bid,
			     dim3	*bDim,
			     dim3	*gDim,
			     int	*wSize)
{
	if (!orig_cudaRegisterFunction) {
		/* Get the pointer to the CUDA-defined register function */
		orig_cudaRegisterFunction = (cudaRegisterFunction_t) dlsym (RTLD_NEXT, "__cudaRegisterFunction");
	}

	std::string kernel (deviceFun);

	/* Hash the function into the kernel map */
	if (kernel_cnt().find (kernel) == kernel_cnt().end()) {
		/* This is a new function. Initialize the entry */
		kernel_cnt()[kernel] = 1;

		/* Keep the host <-> device function mapping in a separate hash */
		kernel_def()[hostFun] = kernel;
	} else {
		/* Update the kernel count */
		kernel_cnt()[kernel] += 1;
	}

	/* Call the cuda register function */
	orig_cudaRegisterFunction (fatCubinHandle, hostFun, deviceFun,
				   deviceName, thread_limit, tid,
				   bid, bDim, gDim,
				   wSize);

	/* Return to caller */
	return;
}
