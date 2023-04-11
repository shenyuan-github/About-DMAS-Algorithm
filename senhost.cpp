#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "xcl2.hpp"

using std::vector;

#define KERNEL_NUM 4
//输入数据总量
static const unsigned int DATA_WIDTH = 128;
static const unsigned int DATA_LENGTH = 3200;
static const unsigned int DATA_SIZE = DATA_LENGTH * DATA_WIDTH;

//输出数据量
static const unsigned int OUT_DATA_SIZE = DATA_LENGTH * DATA_WIDTH/KERNEL_NUM;
//单核心计算量
static const unsigned int single_row =  DATA_LENGTH / KERNEL_NUM;

#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31)};


static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";
//READ bin file ，WRIIte bin file
extern "C"{

int getData(char *fileName, float **data_buffer, size_t *data_size) {
	FILE *data_file;
	long error;
	printf("getdata");
	//data_file = fopen(fileName, "rb");
	printf("%s\n",fileName);
	if ((data_file =fopen(fileName, "rb")) == NULL)
	{
	printf("open fail errno = %d reason = %s \n", errno, strerror(errno));
	}

	if (!data_file) {
		printf("Open data file failed!\n");
		return -1;
	}
	fseek(data_file, 0, SEEK_END);
	*data_size = ftell(data_file);
	fseek(data_file, 0, SEEK_SET);
	//printf("buffdata");
	//memset((*data_buffer), 0, *data_size);
	error = fread(*data_buffer, sizeof(float),*data_size / sizeof(float), data_file);
	if (error != *data_size / sizeof(float)) {
		printf("Read data file failed!\n");
		return -1;
	}
	fclose(data_file);
	return 0;
}

int saveResult(char *fileName, float *data) {
	FILE *writeFile;
	cl_int error = 0;
	writeFile = fopen(fileName, "wb");
	if (!writeFile) {
		printf("Open the file to write failed!\n");
		return -1;
	}

	error = fwrite(data, sizeof(float), OUT_DATA_SIZE, writeFile);  //7600*128=datanum

	if (error != OUT_DATA_SIZE) {
		printf("Write data failed!\n");
		return -1;
	}
	fclose(writeFile);
	return 0;
}
}
// This example illustrates the very simple OpenCL example that performs
// an addition on two vectors
int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string binaryFile = argv[1];
  size_t  data_num;
  // compute the size of array in bytes
  size_t input_datasize= DATA_SIZE * sizeof(float);
  size_t output_datasize= OUT_DATA_SIZE * sizeof(float);
  cl_int err;
  cl::CommandQueue q;
  std::string krnl_name = "pdmas_flow";
  std::vector<cl::Kernel> krnls(KERNEL_NUM);
  cl::Context context;
  cl::Program program;

  //创建 host 端的变量
  vector<float, aligned_allocator<float>> input1(DATA_SIZE);        //数据输入
  //vector<float, aligned_allocator<float>> input2(DATA_SIZE);        //数据输入
  vector<float, aligned_allocator<float>> output_dmas[KERNEL_NUM]; //输出，此处声明已经带有地址偏移，故内核输出无需考虑地址变化，连续输出即可
  for (int i = 0; i < KERNEL_NUM; i++) {
	  output_dmas[i].resize(output_datasize);
  }

  //读取数据
  //for (int i = 0; i < KERNEL_NUM; i++) {
	  float *pInput1=input1.data();  //input get data
	  err = getData("/home/qxm/workspace/kernel_num_4/src/v_3200.bin", &(pInput1), &data_num);
	  //float *pInput2=input2.data();  //input get data
	  //err = getData("/home/qxm/workspace/test1/src/v_3200.bin", &(pInput2), &data_num);
	  if (err != 0) {
		  printf("Get data failed!\n");
		  return -1;
	  }
  //}

  //==================================================================================
  //===================== OPENCL HOST CODE AREA START ================================
  // The get_xil_devices will return vector of Xilinx Devices
  auto devices = xcl::get_xil_devices();

  // read_binary_file() is a utility API which will load the binaryFile and will return the pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  bool valid_device = false;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

    std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    program = cl::Program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    }
    else {
      std::cout << "Device[" << i << "]: program successful!\n";

      // 建立内核
      for (int i = 0; i < KERNEL_NUM; i++) {
    	  std::string cu_id = std::to_string(i + 1);
          std::string krnl_name_full = krnl_name + ":{" + "pdmas_flow_" + cu_id + "}";
          std::cout << "Creating a kernel [" << krnl_name_full.c_str() << "] for CU(" << i + 1 << ")\n";

          // Here Kernel object is created by specifying kernel name along with compute unit.
          // For such case, this kernel object can only access the specific Compute unit
          OCL_CHECK(err, krnls[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
      }

      valid_device = true;
      break; // we break because we found a valid device
    }
  }
  if (!valid_device) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  //=======================================================================================
  //================================= 多核心混合 ============================================
  //核心接口位置(控制连接哪个HBM）
  std::vector<cl_mem_ext_ptr_t> inBufExt1(KERNEL_NUM);
  //std::vector<cl_mem_ext_ptr_t> inBufExt2(KERNEL_NUM);
  std::vector<cl_mem_ext_ptr_t> outBufExt(KERNEL_NUM);
  //核心变量(接口）
  std::vector<cl::Buffer> buffer_input1(KERNEL_NUM);
  //std::vector<cl::Buffer> buffer_input2(KERNEL_NUM);
  std::vector<cl::Buffer> buffer_output(KERNEL_NUM);


  //使用 cl_mem_ext_ptr_t 将 buffer 分配给不同的 HBM
  for (int i = 0; i < KERNEL_NUM; i++) {
    inBufExt1[i].obj = input1.data();
    inBufExt1[i].param = 0;
    inBufExt1[i].flags = bank[i];

    /*inBufExt2[i].obj = input2.data();
    inBufExt2[i].param = 0;
    inBufExt2[i].flags = bank[0];*/

    outBufExt[i].obj = output_dmas[i].data();
    outBufExt[i].param = 0;
    outBufExt[i].flags = bank[i];
  }

  //创建FPGA上的缓存接口
  for (int i = 0; i < KERNEL_NUM; i++) {
    OCL_CHECK(err, buffer_input1[i] = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, input_datasize, &inBufExt1[i], &err));
    //OCL_CHECK(err, buffer_input2[i] = cl::Buffer( context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, input_datasize, &inBufExt2[i], &err));
    OCL_CHECK(err, buffer_output[i] = cl::Buffer( context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR, output_datasize, &outBufExt[i], &err));
  }

  //将输入copy到目标设备的 DDR RAM
  for (int i = 0; i < KERNEL_NUM; i++) {
	  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_input1[i]}, 0 )); /* 0 means from host*/
    //OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_input1[i],buffer_input2[i]}, 0 )); /* 0 means from host*/
  }
  q.finish();

  double kernel_time_in_sec = 0;
  std::chrono::duration<double> kernel_time(0);
  auto kernel_start = std::chrono::high_resolution_clock::now();

  //设置内核参数

  for (int i = 0; i < KERNEL_NUM; i++) {
    // Setting the Arguments
    OCL_CHECK(err, err = krnls[i].setArg(0, buffer_input1[i]));
    //OCL_CHECK(err, err = krnls[i].setArg(argc++, buffer_input2[i]));
    OCL_CHECK(err, err = krnls[i].setArg(1, buffer_output[i]));
    OCL_CHECK(err, err = krnls[i].setArg(2, i*single_row ));
    // Invoking the kernel
    OCL_CHECK(err, err = q.enqueueTask(krnls[i]));
  }
  q.finish();

  auto kernel_end = std::chrono::high_resolution_clock::now();
  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
  kernel_time_in_sec = kernel_time.count();

  // Copy Result from Device Global Memory to Host Local Memory
  for (int i = 0; i < KERNEL_NUM; i++) {
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {buffer_output[i]}, CL_MIGRATE_MEM_OBJECT_HOST) );
  }
  q.finish();

  std::cout << "Kernel run finished!\n";
  std::cout << "Duration time = " << kernel_time_in_sec << " s" << std::endl;
  //=======================================================================================
  //=======================================================================================

  char res_file[60];
  for(int i=0;i<KERNEL_NUM;i++){
 	 sprintf( res_file, "/home/qxm/workspace/kernel_num_4/src/output_pdmas_%d.bin", i);
 	 err = saveResult(res_file, output_dmas[i].data());
 	 //err = saveResult("/home/qxm/workspace/test1/src/output_pdmas_1.bin", output_dmas[i].data());
  }

  std::cout  << "Launch " <<  "PASSED" << std::endl;
  return 0;
}
