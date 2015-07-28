import numpy
import six

from chainer import cuda


def get_conv_outsize(size, k, s, p, cover_all=False):
    if cover_all:
        return (size + p * 2 - k + s - 1) // s + 1
    else:
        return (size + p * 2 - k) // s + 1


def _calc_pad(size, x1, x2):
    left_pad = -x1 if x1 < 0 else 0
    right_pad = x2 - size if x2 > size else 0
    return (left_pad, right_pad)


def _cut_and_pad(img, y1, y2, x1, x2, pval=0):
    """Cut and pad a region (x1, y1)-(x2, y2) from an image.

    """
    assert y1 < y2 and x1 < x2
    ph1, ph2 = _calc_pad(img.shape[2], y1, y2)
    pw1, pw2 = _calc_pad(img.shape[3], x1, x2)
    y1 += ph1
    y2 += ph1
    x1 += pw1
    x2 += pw1

    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph1, ph2), (pw1, pw2)),
                    mode='constant', constant_values=(pval,))
    return img[:, :, y1:y2, x1:x2]


def im2col_cpu(img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False):
    n, c, h, w = img.shape
    out_h = get_conv_outsize(h, kh, sy, ph, cover_all)
    out_w = get_conv_outsize(w, kw, sx, pw, cover_all)

    img = _cut_and_pad(img, -ph, h + ph + sy - 1, -pw, w + pw + sx - 1, pval)

    if ph < 0:
        img[:, :, h + ph * 2:, :] = pval
    if pw < 0:
        img[:, :, :, w + pw * 2:] = pval

    col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    for i in six.moves.range(kh):
        i_lim = i + sy * out_h
        for j in six.moves.range(kw):
            j_lim = j + sx * out_w
            col[:, :, i, j, :, :] = img[:, :, i:i_lim:sy, j:j_lim:sx]

    return col


def im2col_gpu(img, kh, kw, sy, sx, ph, pw, cover_all=False):
    n, c, h, w = img.shape
    out_h = get_conv_outsize(h, kh, sy, ph, cover_all)
    out_w = get_conv_outsize(w, kw, sx, pw, cover_all)

    col = cuda.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
    cuda.elementwise(
        '''
           float* col, const float* img, int h, int w,
           int out_h, int out_w, int kh, int kw, int sy, int sx, int ph, int pw
        ''', '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;

           int in_y = ky + out_y * sy - ph;
           int in_x = kx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h + min(0, ph) &&
               in_x >= 0 && in_x < w + min(0, pw)) {
             col[i] = img[in_x + w * (in_y + h * c0)];
           } else {
             col[i] = 0;
           }
        ''',
        'im2col')(col, img, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw)
    return col


def col2im_cpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape

    img = numpy.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                      dtype=col.dtype)
    for i in six.moves.range(kh):
        i_lim = i + sy * out_h
        for j in six.moves.range(kw):
            j_lim = j + sx * out_w
            img[:, :, i:i_lim:sy, j:j_lim:sx] += col[:, :, i, j, :, :]

    return _cut_and_pad(img, ph, h + ph, pw, w + pw)


def col2im_gpu(col, sy, sx, ph, pw, h, w):
    n, c, kh, kw, out_h, out_w = col.shape

    img = cuda.empty((n, c, h, w), dtype=col.dtype)
    cuda.elementwise(
        '''
           float* img, const float* col, int h, int w,
           int out_h, int out_w, int kh, int kw, int sy, int sx, int ph, int pw
        ''', '''
           int c0 = i / (h * w);
           int y  = i / w % h + ph;
           int x  = i % w + pw;

           int out_y_0 = max(0,     (y - kh + sy) / sy);
           int out_y_1 = min(out_h, (y      + sy) / sy);
           int out_x_0 = max(0,     (x - kw + sx) / sx);
           int out_x_1 = min(out_w, (x      + sx) / sx);

           float val = 0;
           for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
             int ky = y - out_y * sy;
             for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
               int kx = x - out_x * sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val += col[out_x + out_w * k];
             }
           }
           img[i] = val;
        ''',
        'col2im')(img, col, h, w, out_h, out_w, kh, kw, sy, sx, ph, pw)
    return img
