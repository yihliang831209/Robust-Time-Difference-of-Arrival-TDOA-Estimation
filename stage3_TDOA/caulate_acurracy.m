clear all;close all;
%%
% this code will calculate the average DOA accuracy from T60=0.0~T60=1.0
% load mat-file with noisy two-channel spectrogram and mask

ACC_matrix = zeros(4,11); 
for T60=0:1:10
    list = dir(['D:\yihliang_博班\ADSP\performance_evaluation\stage2_create_spectrogram_Mask\traget_T60=0_cleanNormalize_modify\created_mat\T60=0.'+string(T60)+'/*.mat']);
    
    num_hit_mask = 0;
    num_hit_noisy= 0;
    num_hit_IRM = 0;
    
    threhold = 5*pi/180; %5度
    for sample_idx=1:1:length(list)
        load([list(sample_idx).folder '/' list(sample_idx).name]);
        S1 = squeeze(spectrogram(1,:,:));
        S2 = squeeze(spectrogram(2,:,:));
        nfft = 512;
        S_ = cat(3,S1,S2);
        cov_matrix = zeros(2,2,nfft/2+1);
        steer_vector = zeros(2,nfft/2+1);
        theta_f = zeros(nfft/2+1,1);
        Eta = squeeze(mask(1,:,:).*mask(2,:,:));
        for f = 1:1:nfft/2+1
            cov_sum = zeros(2,2);
            for t=1:1:size(S1,2)
                y = squeeze(S_(f,t,:));
                y_H = y';
                cov = y*y_H;
                cov_sum = cov_sum+Eta(f,t)*cov;
            end
            cov_matrix(:,:,f) = cov_sum/sum(Eta(f,:));
            [V,D] = eigs(squeeze(cov_matrix(:,:,f)));
            steer_vector(:,f) = V(:,1);
            % normalize steer vector, makes first entry is real
            angle1 = angle(steer_vector(1,f));
            steer_vector(:,f) = steer_vector(:,f).*exp(1i*-1*angle1);
            theta_f(f) = angle(steer_vector(2,f));
        end
        %%calculate Sim
        Sim = 0;
        TD_max = 0.2/340; % the posible max time delay
        i=1;
        gama = sum(Eta,2);
        for angle_=0.5:-0.5/500:-0.5
%         for tu=TD_max:-TD_max/18:-TD_max
            tu = 0.2*sin(angle_*pi)/340;
            temp_sum=0;
            for f = 1:1:nfft/2+1
                temp_sum = temp_sum + gama(f)*cos(theta_f(f,1)-(-2*pi*(f-1)*(16000/512)*tu));
            end
            Sim(i,1) = tu;
            Sim(i,2) = temp_sum;
            i = i+1;
        end
        % figure()
        % plot(Sim(:,2))
        [M,I] = max(Sim(:,2));
        tu_opti = Sim(I,1);
        DOA = -1*asin(tu_opti*340/0.2);
        if abs(DOA-theta)<=threhold
            num_hit_mask = num_hit_mask+1;
        end
        
        %     direction =-1*asin(Sim(:,1).*340./0.2);
        %     figure()
        %     plot(direction,Sim(:,2));
        %     target_idx =(theta-direction(1,1))/(pi/size(direction,1))+1;
        %     hold on;
        %     plot(theta,M,'*');
        %     legend('predict angle distribution','target')
        %     title('TDOA with predict mask')
        %%
        Eta = ones([257,size(S1,2)]);
        for f = 1:1:nfft/2+1
            cov_sum = zeros(2,2);
            for t=1:1:size(S1,2)
                y = squeeze(S_(f,t,:));
                y_H = y';
                cov = y*y_H;
                cov_sum = cov_sum+Eta(f,t)*cov;
            end
            cov_matrix(:,:,f) = cov_sum/sum(Eta(f,:));
            [V,D] = eigs(squeeze(cov_matrix(:,:,f)));
            steer_vector(:,f) = V(:,1);
            % normalize steer vector, makes first entry is real
            angle1 = angle(steer_vector(1,f));
            steer_vector(:,f) = steer_vector(:,f).*exp(1i*-1*angle1);
            theta_f(f) = angle(steer_vector(2,f));
        end
        %%calculate Sim
        Sim = 0;
        TD_max = 0.2/340; % the posible max time delay
        i=1;
        gama = sum(Eta,2);
        for angle_=0.5:-0.5/500:-0.5
%         for tu=TD_max:-TD_max/18:-TD_max
            tu = 0.2*sin(angle_*pi)/340;
            temp_sum=0;
            for f = 1:1:nfft/2+1
                temp_sum = temp_sum + gama(f)*cos(theta_f(f,1)-(-2*pi*(f-1)*(16000/512)*tu));
            end
            Sim(i,1) = tu;
            Sim(i,2) = temp_sum;
            i = i+1;
        end
        % figure()
        % plot(Sim(:,2))
        [M,I] = max(Sim(:,2));
        tu_opti = Sim(I,1);
        DOA = -1*asin(tu_opti*340/0.2);
        if abs(DOA-theta)<=threhold
            num_hit_noisy = num_hit_noisy+1;
        end
        %     direction =-1*asin(Sim(:,1).*340./0.2);
        %     figure()
        %     plot(direction,Sim(:,2));
        %     target_idx =(theta-direction(1,1))/(pi/size(direction,1))+1;
        %     hold on;
        %     plot(theta,M,'*');
        %     legend('predict angle distribution','target')
        %     title('TDOA without predict mask')
        %%
        decision = IRM<1;
        IRM = IRM.*double(decision)+1.*(1-double(decision));
        Eta = squeeze(IRM(1,:,:).*IRM(2,:,:));
        for f = 1:1:nfft/2+1
            cov_sum = zeros(2,2);
            for t=1:1:size(S1,2)
                y = squeeze(S_(f,t,:));
                y_H = y';
                cov = y*y_H;
                cov_sum = cov_sum+Eta(f,t)*cov;
            end
            cov_matrix(:,:,f) = cov_sum/sum(Eta(f,:));
            [V,D] = eigs(squeeze(cov_matrix(:,:,f)));
            steer_vector(:,f) = V(:,1);
            % normalize steer vector, makes first entry is real
            angle1 = angle(steer_vector(1,f));
            steer_vector(:,f) = steer_vector(:,f).*exp(1i*-1*angle1);
            theta_f(f) = angle(steer_vector(2,f));
        end
        %%calculate Sim
        Sim = 0;
        TD_max = 0.2/340; % the posible max time delay
        i=1;
        gama = sum(Eta,2);
        for angle_=0.5:-0.5/500:-0.5
%         for tu=TD_max:-TD_max/18:-TD_max
            tu = 0.2*sin(angle_*pi)/340;
            temp_sum=0;
            for f = 1:1:nfft/2+1
                temp_sum = temp_sum + gama(f)*cos(theta_f(f,1)-(-2*pi*(f-1)*(16000/512)*tu));
            end
            Sim(i,1) = tu;
            Sim(i,2) = temp_sum;
            i = i+1;
        end
        % figure()
        % plot(Sim(:,2))
        [M,I] = max(Sim(:,2));
        tu_opti = Sim(I,1);
        DOA = -1*asin(tu_opti*340/0.2);
        if abs(DOA-theta)<=threhold
            num_hit_IRM = num_hit_IRM+1;
        end
        %     direction =-1*asin(Sim(:,1).*340./0.2);
        %     figure()
        %     plot(direction,Sim(:,2));
        %     target_idx =(theta-direction(1,1))/(pi/size(direction,1))+1;
        %     hold on;
        %     plot(theta,M,'*');
        %
        %     legend('predict angle distribution','target')
        %     title('TDOA with ideal mask')
        disp('process T60=0.'+ string(T60))
        disp('process '+string(sample_idx)+' out of '+string(length(list)))
        disp('ACC_mask: '+string(num_hit_mask/sample_idx))
        disp('ACC_noisy: '+string(num_hit_noisy/sample_idx))
        disp('ACC_IRM: '+string(num_hit_IRM/sample_idx))
        ACC_matrix(1,T60+1) = T60/10;
        ACC_matrix(2,T60+1) = num_hit_mask/sample_idx;
        ACC_matrix(3,T60+1) =num_hit_noisy/sample_idx;
        ACC_matrix(4,T60+1) = num_hit_IRM/sample_idx;
    end
end
save('target_T60=0_cleanWNormalize_modify_sampleOnAngle.mat','ACC_matrix')
