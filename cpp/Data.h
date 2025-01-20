#ifndef Stars_Data_h
#define Stars_Data_h

#include <vector>

namespace Stars
{

class Data
{
    private:
        std::vector<double> x, y, v, verr;

    public:
        Data(const char* filename);

        static Data instance;
};

} // namespace

#endif
