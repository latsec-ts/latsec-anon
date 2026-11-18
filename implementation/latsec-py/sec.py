"""SEC: dealing, reconstruction vectors and the coefficient checks.

Section 2.4 of the paper, and the SEC paper [46]:

  * SEC.Deal_pp(s), Eq. (1): S = G [s^T; R] = g s^T + G_hat R, so party i holds
    s_i = g_i s + sum_j G_hat[i][j] R_j.
  * SEC.Rec_pp(T, (x_i)) = sum_i lambda_{T,i} x_i for a vector lambda_T with
    lambda_T^T G_T = e_1^T (Item 2).
  * Public coefficient checks (Item 3): lambda^T G_T = e_1^T, |lambda|_inf <=
    beta_lambda and wt(lambda) <= omega_lambda are exact integer checks.

reconstruction_vector() is [46, Algorithm 1] with the conventions of the
profile identifier "xlatsec-recon-v1": an exact row HNF gives a witness and
the saturated left kernel K_T; an exact integral LLL (Cohen, Algorithm 2.6.7)
reduces K_T; Babai's nearest plane and a greedy l_inf refinement shorten the
witness.  Every tie rule below is part of that identifier: two implementations
that follow them return the same vector.  Under the paper's predicate
the vector is carried in the session message, so other implementations may use
any algorithm whose output passes the checks.

Party indices are zero-based: party i of the paper is index i - 1.
"""


# --- SEC.Deal ------------------------------------------------------------------

def deal(g_rows, secret, sample_mask_poly, rng):
    """Shares s_i = g_i s + sum_j G_hat[i][j] R_j of a secret s in R^ell.

    g_rows: the rows of G = [g | G_hat].  The k - 1 rows of R are sampled one
    after another, each as ell polynomials of D_{sigma_R} coefficients.
    """
    ell = len(secret)
    k = len(g_rows[0])
    masks = [[sample_mask_poly(rng) for _ in range(ell)] for _ in range(k - 1)]
    shares = []
    for row in g_rows:
        share = [[row[0] * x for x in poly] for poly in secret]
        for j, mask in enumerate(masks):
            if row[j + 1] != 0:
                share = [[x + row[j + 1] * y for x, y in zip(sp, mp)] for sp, mp in zip(share, mask)]
        shares.append(share)
    return shares


def combine_z(lam, items):
    """sum_i lam_i v_i over the integers, for vectors of polynomials."""
    out = [[0] * len(poly) for poly in items[0]]
    for coeff, item in zip(lam, items):
        if coeff:
            out = [[x + coeff * y for x, y in zip(op, ip)] for op, ip in zip(out, item)]
    return out


def combine_q(lam, items, q):
    """sum_i lam_i v_i mod q, for vectors of residue polynomials."""
    return [[x % q for x in poly] for poly in combine_z(lam, items)]


# --- Public coefficient checks (Section 2.4, Item 3) ---------------------------

def is_reconstruction_vector(g_rows, act, lam):
    """lambda^T G_act = e_1^T."""
    if len(lam) != len(act):
        return False
    k = len(g_rows[0])
    for col in range(k):
        total = sum(c * g_rows[i][col] for c, i in zip(lam, act))
        if total != (1 if col == 0 else 0):
            return False
    return True


# --- Exact integer linear algebra ------------------------------------------------

def round_half_up(num, den):
    """floor(num / den + 1/2) for den > 0: nearest integer, ties toward +inf."""
    return (2 * num + den) // (2 * den)


def exact_div(num, den):
    q, r = divmod(num, den)
    assert r == 0, "inexact division in integral Gram-Schmidt"
    return q


def hnf_in_place(rows, main_cols):
    """Row Hermite normal form on the first main_cols columns, in place.

    Row operations act on whole rows, so trailing columns carry the
    transformation.  In each column the remaining row with the smallest
    nonzero |entry| (first on ties) becomes the pivot row, and the others are
    reduced by it with nearest-integer quotients (ties toward +inf), until one
    nonzero entry is left.  The pivot is made positive, and the entries above it
    are reduced into [0, pivot).  Returns the pivot columns.
    """
    nrows = len(rows)
    pivots, rank = [], 0
    for col in range(main_cols):
        if rank == nrows:
            break
        while True:
            best, others = None, False
            for i in range(rank, nrows):
                if rows[i][col] == 0:
                    continue
                if best is None:
                    best = i
                else:
                    others = True
                    if abs(rows[i][col]) < abs(rows[best][col]):
                        best = i
            if best is None:
                break
            rows[rank], rows[best] = rows[best], rows[rank]
            if not others:
                break
            pivot_row = rows[rank]
            pivot_abs = abs(pivot_row[col])
            for i in range(rank + 1, nrows):
                if rows[i][col] == 0:
                    continue
                quot = round_half_up(rows[i][col], pivot_abs)
                if pivot_row[col] < 0:
                    quot = -quot
                rows[i] = [x - quot * y for x, y in zip(rows[i], pivot_row)]
        if rows[rank][col] == 0:
            continue
        if rows[rank][col] < 0:
            rows[rank] = [-x for x in rows[rank]]
        pivot_row = rows[rank]
        for i in range(rank):
            quot = rows[i][col] // pivot_row[col]
            rows[i] = [x - quot * y for x, y in zip(rows[i], pivot_row)]
        pivots.append(col)
        rank += 1
    return pivots


def row_hnf(matrix):
    """H = U A with U unimodular: returns (H rows, U rows, pivots)."""
    nrows, ncols = len(matrix), len(matrix[0])
    rows = [list(r) + [1 if i == j else 0 for j in range(nrows)] for i, r in enumerate(matrix)]
    pivots = hnf_in_place(rows, ncols)
    return [r[:ncols] for r in rows], [r[ncols:] for r in rows], pivots


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class Gso:
    """Integral Gram-Schmidt data (Cohen, Section 2.6.3): d[0] = 1,
    d[i] = prod_{j <= i} |b*_j|^2, and lam[i][j] = d[j+1] mu_{i,j}."""

    def __init__(self, r):
        self.d = [1] * (r + 1)
        self.lam = [[0] * r for _ in range(r)]

    def extend(self, basis, k):
        for j in range(k + 1):
            u = dot(basis[k], basis[j])
            for i in range(j):
                u = exact_div(self.d[i + 1] * u - self.lam[k][i] * self.lam[j][i], self.d[i])
            if j < k:
                self.lam[k][j] = u
            else:
                assert u != 0, "basis vectors are linearly dependent"
                self.d[k + 1] = u


def integral_gso(basis):
    gso = Gso(len(basis))
    for k in range(len(basis)):
        gso.extend(basis, k)
    return gso


def size_reduce(basis, gso, k, l):
    """Cohen's REDI(k, l): nothing if |mu_{k,l}| <= 1/2, otherwise subtract
    floor(mu_{k,l} + 1/2) b_l from b_k."""
    d_l = gso.d[l + 1]
    if abs(2 * gso.lam[k][l]) <= abs(d_l):
        return
    quot = round_half_up(gso.lam[k][l], d_l)
    basis[k] = [x - quot * y for x, y in zip(basis[k], basis[l])]
    gso.lam[k][l] -= quot * d_l
    for i in range(l):
        gso.lam[k][i] -= quot * gso.lam[l][i]


def swap_adjacent(basis, gso, k, kmax):
    """Cohen's SWAPI(k): exchanges b_k and b_{k-1}."""
    basis[k], basis[k - 1] = basis[k - 1], basis[k]
    for j in range(k - 1):
        gso.lam[k - 1][j], gso.lam[k][j] = gso.lam[k][j], gso.lam[k - 1][j]
    lam = gso.lam[k][k - 1]
    b = exact_div(gso.d[k - 1] * gso.d[k + 1] + lam * lam, gso.d[k])
    for i in range(k + 1, kmax + 1):
        t = gso.lam[i][k]
        new_ik = exact_div(gso.d[k + 1] * gso.lam[i][k - 1] - lam * t, gso.d[k])
        gso.lam[i][k - 1] = exact_div(b * t + lam * new_ik, gso.d[k + 1])
        gso.lam[i][k] = new_ik
    gso.d[k] = b


def lll_reduce(basis, delta_num, delta_den):
    """Cohen's integral LLL (Algorithm 2.6.7), in place, with delta = num/den.

    k starts at the second vector.  b_k is size-reduced against b_{k-1}; the
    Lovasz condition is den (d_k d_{k-2} + lam_{k,k-1}^2) >= num d_{k-1}^2.  On
    failure b_k and b_{k-1} swap and k decreases (not below the second vector);
    otherwise b_k is size-reduced against b_{k-2}, ..., b_1 and k increases.
    """
    r = len(basis)
    gso = Gso(r)
    if r == 0:
        return gso
    gso.extend(basis, 0)
    k, kmax = 1, 0
    while k < r:
        if k > kmax:
            kmax = k
            gso.extend(basis, k)
        while True:
            size_reduce(basis, gso, k, k - 1)
            lam = gso.lam[k][k - 1]
            lhs = delta_den * (gso.d[k + 1] * gso.d[k - 1] + lam * lam)
            rhs = delta_num * gso.d[k] * gso.d[k]
            if lhs < rhs:
                swap_adjacent(basis, gso, k, kmax)
                if k > 1:
                    k -= 1
            else:
                for l in range(k - 2, -1, -1):
                    size_reduce(basis, gso, k, l)
                k += 1
                break
    return gso


def babai_nearest_plane(basis, gso, target):
    """target reduced modulo the lattice of `basis`: planes from the last basis
    vector to the first, coefficients rounded with ties toward +inf."""
    r = len(basis)
    w = list(target)
    nu = []
    for j in range(r):
        u = dot(w, basis[j])
        for i in range(j):
            u = exact_div(gso.d[i + 1] * u - nu[i] * gso.lam[j][i], gso.d[i])
        nu.append(u)
    for i in range(r - 1, -1, -1):
        c = round_half_up(nu[i], gso.d[i + 1])
        if c == 0:
            continue
        w = [x - c * y for x, y in zip(w, basis[i])]
        for j in range(i):
            nu[j] -= c * gso.lam[i][j]
    return w


def linf(v):
    return max((abs(x) for x in v), default=0)


def greedy_reduce_linf(lam, basis, passes, window):
    """Greedy l_inf refinement ([46, Algorithm 1], second stage).

    For each basis vector b in order: target the coordinate j of largest |lam_j|
    (smallest index on ties), scan alpha in [alpha0 - window, alpha0 + window]
    with alpha0 = round(-lam_j / b_j), and keep the first candidate lam + alpha b
    of smallest l_inf norm if it strictly improves.  Stop after `passes` passes
    or a pass without improvement.
    """
    lam = list(lam)
    if not basis or not lam:
        return lam
    best = linf(lam)
    for _ in range(passes):
        improved = False
        for b in basis:
            j = 0
            for idx, x in enumerate(lam):
                if abs(x) > abs(lam[j]):
                    j = idx
            if b[j] == 0:
                continue
            num, den = (lam[j], -b[j]) if b[j] < 0 else (-lam[j], b[j])
            alpha0 = round_half_up(num, den)
            norm_best, lam_best = best, None
            for alpha in range(alpha0 - window, alpha0 + window + 1):
                cand = [x + alpha * y for x, y in zip(lam, b)]
                cn = linf(cand)
                if cn < norm_best:
                    norm_best, lam_best = cn, cand
            if lam_best is not None:
                lam, best, improved = lam_best, norm_best, True
        if not improved:
            break
    return lam


def reconstruction_vector(g_t, delta_num=99, delta_den=100, passes=2, window=2):
    """[46, Algorithm 1] for the rows g_t of G: a short lambda with
    lambda^T g_t = e_1^T, or None if the set is not authorized."""
    if not g_t or not g_t[0]:
        return None
    ncols = len(g_t[0])
    target = [1] + [0] * (ncols - 1)
    hnf, transform, pivots = row_hnf(g_t)
    rank = len(pivots)
    if rank == 0 or hnf[0] != target:
        return None
    witness = transform[0]
    kernel = [list(r) for r in transform[rank:]]
    pivots = hnf_in_place(kernel, len(g_t))
    kernel = kernel[:len(pivots)]
    if len(kernel) > 1:
        gso = lll_reduce(kernel, delta_num, delta_den)
    else:
        gso = integral_gso(kernel)
    after_babai = babai_nearest_plane(kernel, gso, witness)
    lam = greedy_reduce_linf(after_babai, kernel, passes, window)
    assert [dot(lam, [row[c] for row in g_t]) for c in range(ncols)] == target
    return lam
