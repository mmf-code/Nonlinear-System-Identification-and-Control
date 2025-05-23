% =========================================================================
% Example usage of System1 class in MATLAB
% Make sure System1.p and is in the current folder or path.
% =========================================================================
clc;clear;
%% --------------------- SYSTEM 1 USAGE ----------------------------------

s2 = System2(42); % Last two digits of student ID (last digit if it is 0x)

% Step response
[~, t2, y2] = s2.step();
figure;
plot(t2, y2);
title('System2 - Step Response');
grid on;

% Ramp response
[~, t2, y2] = s2.ramp();
figure;
plot(t2, y2);
title('System2 - Ramp Response');
grid on;

% Arbitrary input sequence
s2 = s2.reset(); % Reset internal state
input_seq = linspace(0, 1, 100);
output_seq = zeros(size(input_seq));

for i = 1:length(input_seq)
    [s2, output_seq(i)] = s2.output(input_seq(i));
end

figure;
plot(input_seq, output_seq);
title('System2 - Arbitrary Input Test');
grid on;