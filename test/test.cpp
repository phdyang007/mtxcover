#include<iostream>

using namespace std;



int main()
{
    for(int i=1;i<10;i++){
        cout<<i<<' '<<endl;
        if(i==4){
            i=1;
        }
    }

    return 0;
}