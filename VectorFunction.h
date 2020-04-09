
enum Modes {
    COMPLEXES_REALS,
    COMPLEXES,
    REALS,
    ONE_COMPLEX
};

const double EPSILON = 1e-10;

class VectorFunction {
private:
    double *vec_r[3];
    fftw_complex *vec_c[3];
    fftw_plan forward_plan[3], backward_plan[3];

    const Modes mode;

    const ptrdiff_t N;
    const double TAU, ETA;
    const ptrdiff_t INDEX_LEFT, INDEX_RIGHT;
    const double LEFT_RANGE, RIGHT_RANGE;

    const ptrdiff_t alloc_local, local_dim0_size, local_dim0_start;

    ptrdiff_t *indexes;
    const double NORMALIZATION_CONSTANT;
    const int rank, size;

    void cross_product(const VectorFunction &, const VectorFunction &);

    void divergence(const VectorFunction &);

    void rotor(const VectorFunction &, const VectorFunction &);

public:
    VectorFunction(Modes, ptrdiff_t, double, double, double, double, ptrdiff_t, ptrdiff_t, ptrdiff_t, int, int);

    virtual ~VectorFunction();

    void forward_transformation();

    void backward_transformation();

    void initialize(std::vector<std::function<double(const double, const double, const double)>>);

    void do_step(const VectorFunction &, VectorFunction &);

    double energy_fourier() const;

    void correction(VectorFunction &);
};
