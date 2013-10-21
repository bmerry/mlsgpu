% mlsgpu: surface reconstruction from point clouds
% Copyright (C) 2013  University of Cape Town
%
% This file is part of mlsgpu.
%
% mlsgpu is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
