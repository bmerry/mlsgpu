global R = 10

function y = phi(x)
    global R;
    y = (1 - (x / R)^2)^4;
end

function y = xphi(x)
    y = x * phi(x);
end

function y = x2phi(x)
    y = x^2 * phi(x);
end

function y = x3phi(x)
    y = x^3 * phi(x);
end

sum_w = quad(@(t) 1, 0, pi) * quad('xphi', 0, R)
sum_wp = quad(@(t) sin(t), 0, pi) * quad('x2phi', 0, R)
sum_wpp = quad(@(t) 1, 0, pi) * quad('x3phi', 0, R)
(sum_wp / sum_w) / sqrt(sum_wpp / sum_w)
2560 * sqrt(6) / (3465 * pi)
