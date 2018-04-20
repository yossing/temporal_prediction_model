%Author: Yosef Singer

function [row, col,maxval] = max2(p)

[maxrows,rowind] = max(p);
% find the best of these maxima
% across the columns
[maxval,col] = max(maxrows);
row = rowind(col);   