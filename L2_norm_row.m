%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
%Authors: Martin Saveski, Amin Mantrach
function Xnorm = L2_norm_row(X)
  Xnorm= spdiags(1 ./ (sqrt(sum(X.*X,2)) + eps),0,size(X,1),size(X,1)) * X;
end
