clear all;close all;
[clean,fs] = audioread('fadg0_1-sa1.wav');

c = 340;                    % Sound velocity (m/s)
fs = 16000;                 % Sample frequency (samples/s)
r1 = [3.9 4 1.5];              % Receiver position [x y z] (m)
r2 = [4.1 4 1.5];              % Receiver position [x y z] (m)

theta =0.2*pi;            % sound source angle [0.5pi~-0.5pi]
Distance = 1;                         % distance between reciver and source
s = [4 4 1.5]+[Distance*-sin(theta) Distance*cos(theta) 0];              % Source position [x y z] (m)
L = [8 8 3];                % Room dimensions [x y z] (m)
beta1 = 0.9;                 % Reverberation time (s)
n = 8000;
h1 = rir_generator(c, fs, r1, s, L, beta1,n);
h2 = rir_generator(c, fs, r2, s, L, beta1,n);
C1 = conv(clean,h1);
C2 = conv(clean,h2);
figure()
plot(C1);
hold on;
plot(C2)
legend('C1','C2')
%% 
save('test.mat','C1','C2','theta')