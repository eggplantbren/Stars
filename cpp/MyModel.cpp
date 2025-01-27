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
    xc = -1.0 + 2.0*rng.rand();
    yc = -1.0 + 2.0*rng.rand();
    q = rng.rand();
    phi = 2.0*M_PI*rng.rand();
    L = pow(10.0, -3.0 + 3.0*rng.rand());

    mu = -1000.0 + 2000.0*rng.rand();
    sigma = pow(10.0, -3.0 + 6.0*rng.rand());

    A = pow(10.0, -3.0 + 6.0*rng.rand());
    phi_v = 2.0*M_PI*rng.rand();
}

double MyModel::perturb(DNest4::RNG& rng)
{
    double logH = 0.0;

    int which = rng.rand_int(9);

    if(which == 0)
    {
        xc += 2.0*rng.randh();
        DNest4::wrap(xc, -1.0, 1.0);
    }
    else if(which == 1)
    {
        yc += 2.0*rng.randh();
        DNest4::wrap(yc, -1.0, 1.0);
    }
    else if(which == 2)
    {
        q += rng.randh();
        DNest4::wrap(q, 0.0, 1.0);
    }
    else if(which == 3)
    {
        phi += 2.0*M_PI*rng.rand();
        DNest4::wrap(phi, 0.0, 2.0*M_PI);
    }
    else if(which == 4)
    {
        L = log10(L);
        L += 3.0*rng.randh();
        DNest4::wrap(L, -3.0, 0.0);
        L = pow(10.0, L);
    }
    else if(which == 5)
    {
        mu += 2000.0*rng.randh();
        DNest4::wrap(mu, -1000.0, 1000.0);
    }
    else if(which == 6)
    {
        sigma = log10(sigma);
        sigma += 6.0*rng.randh();
        DNest4::wrap(sigma, -3.0, 3.0);
        sigma = pow(10.0, sigma);
    }
    else if(which == 7)
    {
        A = log10(A);
        A += 6.0*rng.randh();
        DNest4::wrap(A, -3.0, 3.0);
        A = pow(10.0, A);
    }
    else
    {
        phi_v += 2.0*M_PI*rng.rand();
        DNest4::wrap(phi_v, 0.0, 2.0*M_PI);
    }



    return logH;
}

double MyModel::log_likelihood() const
{
    double logL = 0.0;

    const Data& data = Data::instance;

    double xx, yy, var;
    double cos_phi = cos(phi);
    double sin_phi = sin(phi);
    for(size_t i=0; i<data.x.size(); ++i)
    {
        // Spatial part
        xx = (data.x[i] - xc)*cos_phi + (data.y[i] - yc)*sin_phi;
        yy = -(data.x[i] - xc)*sin_phi + (data.y[i] - yc)*cos_phi;
        logL += -0.5*log(2.0*M_PI*L*L)
                - 0.5*(pow(xx, 2)*q + pow(yy, 2)/q)/(L*L);

        // Kinematic part
        // Predicted value of radial velocity from the parameters
        double theta = atan2(data.y[i] - yc, data.x[i] - xc);
        double mu_v = mu + A*sin(theta - phi_v);

        var = sigma*sigma + data.verr[i]*data.verr[i];
        logL += -0.5*log(2.0*M_PI*var)
                - 0.5*pow(data.v[i] - mu_v, 2)/var;
    }

    return logL;
}

void MyModel::print(std::ostream& out) const
{
    out << xc << ' ' << yc << ' ' << q << ' ' << phi << ' ' << L << ' ';
    out << mu << ' ' << sigma << ' ';
    out << A << ' ' << phi_v;
}

std::string MyModel::description() const
{
    return std::string("xc yc q phi L mu sigma A phi_v");
}

} // namespace
