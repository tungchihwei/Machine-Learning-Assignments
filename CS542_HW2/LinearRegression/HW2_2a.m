table = load( 'detroit.mat' );
 
HOM = table.data(:, 10);
FTP = table.data(:, 1);
WE = table.data(:, 9);

first = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 1];
matrix = [first, FTP, WE];
error = [] ;

for i = 2:8
     matrix_tmp = table.data(:, i);
     matrix2 = [matrix, matrix_tmp];
     
     dif = (matrix2*((((matrix2')*matrix2)^(-1))*(matrix2')*HOM))-HOM;
     sum_error = sum((dif.^2));
     
     lowest_error = sum_error/(2*13);
     error = [error; lowest_error];
end

val = min(error(~ismember(error,0)));
third = find(val==error);
if third == 1
    disp("The third variable in determining HOM is UEMP");
elseif third == 2
    disp("The third variable in determining HOM is MAN");
elseif third == 3
    disp("The third variable in determining HOM is LIC");
elseif third == 4
    disp("The third variable in determining HOM is GR");
elseif third == 5
    disp("The third variable in determining HOM is NMAN");
elseif third == 6
    disp("The third variable in determining HOM is GOV");
elseif third == 7
    disp("The third variable in determining HOM is HE");
end

plot(error);
title('Errors of variables');
xlabel('Variables');
ylabel('Errors');
 