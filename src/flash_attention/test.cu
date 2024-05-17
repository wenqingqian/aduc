#include <vector>
#include <iostream>
#include <fstream>
extern std::vector<float> flash_attention_forward(const int N, const int d);

int main(){
	std::ofstream outputFile("output.txt");
	if (!outputFile.is_open()) {
		std::cerr << "Failed to open the file." << std::endl;
		return 1;
	}
	outputFile << "[\"N(seq_len)\", \"d(hidden_dim)\", \"Nxd\", \"time\", \"Type\"],\n";
	
	for(int i = 0; i<100; i ++){
		flash_attention_forward(32, 32);
	}
	for(int N = 1; N < 128; N ++){
		for(int d = 1; d < 80; d ++){
			printf("N:%d d:%d\n", N, d);
			std::vector<float> time = flash_attention_forward(32*N,d);
			outputFile << "[" << N*32 << ", " << d << ", " << N*32*d << ", " << time[0] << ", \"gpu_flash_attnv1\"],\n";
			outputFile << "[" << N*32 << ", " << d << ", " << N*32*d << ", " << time[1] << ", \"gpu_flash_attnv2\"],\n";
			outputFile << "[" << N*32 << ", " << d << ", " << N*32*d << ", " << time[2] << ", \"cpu_standard_attn\"],\n";
			outputFile << "[" << N*32 << ", " << d << ", " << N*32*d << ", " << time[3] << ", \"cpu_flash_attn\"],\n";

			outputFile.flush();
		}
	}
	outputFile.close();
}