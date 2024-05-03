#include <iostream>

using namespace std;

template <int ff = 1>
void func(){
	if constexpr (ff-- == 1)
		cout<<ff<<endl;
	else
		cout<<-1<<endl;
}

int main(){
	func();
}