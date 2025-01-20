#include "MyModel.h"
#include "DNest4/code/DNest4.h"
#include <cmath>
#include "Data.h"

namespace Stars
{

MyModel::MyModel()
{

}

void MyModel::from_prior(DNest4::RNG& rng)
{
    mu = -1000.0 + 2000.0*rng.rand();
    sigma = pow(10.0, -3.0 + 6.0*rng.rand());
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(2);
    if(which == 0)
    {
        mu += 2000.0*rng.randh();
        DNest4::wrap(mu, -1000.0, 1000.0);
    }
    else
    {
        sigma = log10(sigma);
        sigma += 6.0*rng.randh();
        DNest4::wrap(sigma, -3.0, 3.0);
        sigma = pow(10.0, sigma);
    }

    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    const Data& data = Data::instance;

    double var;
    for(size_t i=0; i<data.x.size(); ++i)
    {
        var = sigma*sigma + data.verr[i]*data.verr[i];
        logL += -0.5*log(2.0*M_PI*var)
                - 0.5*pow(data.v[i] - mu, 2)/var;
    }

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    out << mu << ' ' << sigma;
}

std::string MyModel::description() const
{
    return std::string("mu sigma");
}

} // namespace
