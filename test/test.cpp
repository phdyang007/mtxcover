#include<iostream>

using namespace std;


void print(int *vec, int length)
{
    for (int i = 0; i< length; i++)
    {
        cout<<vec[i]<<' ';
    }
}

int main()
{
    int *a = new int [8];
    for (int i =0; i<8;i++)
    {
        a[i]=i;
    }

    print(a+3, 3);
    

    delete [] a;

    return 0;
}