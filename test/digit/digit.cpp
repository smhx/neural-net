#include "../tester.h"
#include <string>
#include <cstdio>
using namespace std;

const int ndigits=10, imgsize=28;

int ntraining, ntest;

// taken from https://compvisionlab.wordpress.com/

inline int revint(int i) {
	unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void readimg(string fname, trbatch& tr) {
	ifstream file(fname, ios::binary);
	int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= revint(magic_number);
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= revint(number_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= revint(n_rows);
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= revint(n_cols);

    printf("n_rows = %d, n_cols = %d\n", n_rows, n_cols);

    tr = trbatch(number_of_images);
    
    for(int i=0;i<number_of_images;++i) {
    	tr[i].first = vdbl(n_rows * n_cols);
        for(int r=0;r<n_rows;++r){
            for(int c=0;c<n_cols;++c){
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                tr[i].first[(n_rows*r)+c]= (double)temp;
            }
        }
    }
}

void readlbl(string fname, trbatch& tr) {
	
    ifstream file (fname, ios::binary);

    int magic_number = 0;
    int number_of_images = 0;
    file.read((char*) &magic_number, sizeof(magic_number));
    magic_number = revint(magic_number);
    file.read((char*) &number_of_images,sizeof(number_of_images));
    number_of_images = revint(number_of_images);

    for(int i = 0; i < number_of_images; ++i){
        unsigned char temp = 0;
        file.read((char*) &temp, sizeof(temp));
        tr[i].second = vdbl(ndigits, 0.0);
        tr[i].second[temp] = 1.0;
    }
	
}


void read(Tester& t) {
	readimg("test/digit/data/trainimg", t.training);
	readimg("test/digit/data/testimg", t.testing);

	readlbl("test/digit/data/trainlbl", t.training);
	readlbl("test/digit/data/testlbl", t.testing);
}

void train(){
	Tester tester;
	read(tester);
	cout << "Finished reading\n";
	tester.sizes = {imgsize*imgsize, 100, ndigits};

	tester.numEpochs = 200;
	tester.batchSize = 20;

	tester.lrate = 100.0;
	tester.maxRate = 150.0;
	tester.minRate = 30.0;
	
	tester.checker = largestCheck;

	tester.write = true;

	tester.train();
}

int main() {
	train();
}