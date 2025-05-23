% =========================================================================
% Example usage of System1 class in MATLAB
% Make sure System1.p and is in the current folder or path.
% =========================================================================
clc;clear;
%% --------------------- SYSTEM 1 USAGE ----------------------------------

s1 = System1(42); % Last two digits of student ID (last digit if it is 0x)

% 1 - Step response 
[t1, y1] = s1.step();

% Step response (custom time vector)
t_vec = linspace(0, 5, 200);
[t1, y1] = s1.step(t_vec);
figure;
plot(t1, y1);
title('System1 - Step Response (Custom Time)');
grid on;

% 2 - Ramp response
[t1, y1] = s1.ramp();

% Ramp response (custom time vector)
t_vec = linspace(0, 5, 200);
[t1, y1] = s1.ramp(t_vec);
figure;
plot(t1, y1);
title('System1 - Ramp Response (Custom Time)');
grid on;

% Arbitrary input output
input_vec = sort(rand(1, 10));
output_vec = s1.output(input_vec);
figure;
plot(input_vec, output_vec, 'o-');
title('System1 - Arbitrary Input Test');
grid on;