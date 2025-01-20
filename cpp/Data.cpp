#include "Data.h"
#include <fstream>
#include <iostream>

namespace Stars
{

Data Data::instance{"../data.txt"};

Data::Data(const char* filename)
{
    std::fstream fin(filename, std::ios::in);
    double _x, _y, _v, _verr;

    int n = 0;
    while(fin >> _x && fin >> _y && fin >> _v && fin >> _verr)
    {
        x.push_back(_x);
        y.push_back(_y);
        v.push_back(_v);
        verr.push_back(_verr);
        ++n;
    }

    fin.close();

    std::cout << "# Loaded " << n << " points from " << filename << '.';
    std::cout << std::endl;
}








} // namespace
