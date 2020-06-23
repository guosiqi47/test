% This simulation study aims to explore the effects of different parameters
% for XLOT reconstruction. We will deal with cylindrical shaped targets in
% this code, which can contain up to 6 sets of targets, each having 1, 3, or
% 6 tube targets. 

% Modified with optical fiber detection.

% Modified 11/15/2018 by Michael Lun to include step size effects

clear; clc;

% Phantom Geometry Parameters (Radius, Height, Scan Depth) (Unit: mm)  
% Size of cylindrical phantom, depth (from top surface) of target
% R = 6.4; height = 10; depth = 5; 
R = 15; height = 40; depth = 20; 
%% Setting up initial parameters. 

% Number of Projections ***
nProj_all =60; % [2 3 6] 
nmask = 1;

% Non linearity of x-ray beam* 
% Perturbation angle of F
F_pert_angle_all = [0] * 180./nProj_all;   % [0 1/4 1/3 1/2];

% X-ray beam width *** (smaller = better resolution) 
% beam_width = [0.4];     % [0.025, 0.05, 0.1, 0.2];

% Noise Level (10/100 = 10% Noise, Gaussian)*
% Simulated noise level ***
noise_level_all = 20/100;   % [10 20 50]/100;

% Regularization Parameter, control how to reduce noise effects in recon. *
% Higher number = less noise*
% Reduces influence of noise*
% Regularization percentage *** 
percentage = 10;     % [0 10 20 40 60];

% How many loops in reconstruction. (Tolerance)* 
% Max iteration number ***
IterNumMax = 20; 

% Flag (True (Use mask) or False (don't use mask))*
% mask = reduce unknown node number.*
% can improve resolution. *
% Reducing recon nodes (a.k.a. XCT mask), "true" or "false"
XCT_mask = false;
threshold_percentage = 0.1; % threshold for measurements with excited fluorophores

%%%%%%%%%%%% More parameters %%%%%%%%%%%%%
% Size of mesh grid (mm) (Smaller = clearer image)*
% Pixel Size ***
pix_size = 0.1; % [0.025]

% Linear Scan Step Number 
% Number of Scans for each projection
% nScan_all = 2 * R./beam_width;
% step_size_red_factor =1; %how many times to reduce the step size (i.e. 4 = 1/4)
% nScan_all = ceil((2 * R./beam_width) * step_size_red_factor); %ceil向上取整

% Mesh grid coordinates, ensures phantom center is center of grid*
[xi,yi,zi] = meshgrid(-R:pix_size:(R-pix_size/2), -R:pix_size:(R-pix_size/2), height-depth);

%% Simulate targets

% Radius of target tubes, unit mm
% D and D_inner must be same size!*
D = [1.5]; D_inner = [0]; %  % 40 * pix_size ; % [0.1, 0.05, 0.025]; 
 
centerAll = cell(max(size(D)),1);

% % 1 center each set
% % % D = 5; D_inner = 4/5 * D; % if target is one circle
% for nr = 1:max(size(D))
% 
%     % generate the initial set of 3 centers for targets.
%     center = [0, -R/2]';
%     
%     % rotate pi/3 degrees to generate more targets of different sizes
%     rot_angle = pi/3*(nr - 1); 
%     center = [cos(rot_angle) -sin(rot_angle); sin(rot_angle), cos(rot_angle)]*center;
%     centerAll{nr} = center;
%     
% end

% % % 3 centers each set
% for nr = 1:max(size(D))
% 
%     % generate the initial set of 3 centers for targets.
% %     center = repmat([0, -R/2]', [1 3]) + D(nr)*[-1, 0, 1; (2-sqrt(3)), 2, 2-sqrt(3)];
%     center = [-1.5 1.5 0 ; -7 -7 -9.5981];
% %      center = [-2.4 2.4 0; -R/2 -R/2  -R/2-D(1)*sqrt(3)];
%     % rotate pi/3 degrees to generate more targets of different sizes
%     rot_angle = pi/3*(nr - 1); 
%     center = [cos(rot_angle) -sin(rot_angle); sin(rot_angle), cos(rot_angle)]*center;
%     centerAll{nr} = center;
% %     
% end

% 4 centers each set
% for nr = 1:max(size(D))
% 
%     % generate the initial set of 3 centers for targets.
%     center = [-3.5; -2.5];
% %     center = [-3.5 -3.5 0 3.5; -2.5 -7.5 -7.5 -7.5];
%     % rotate pi/3 degrees to generate more targets of different sizes
%     rot_angle = pi/3*(nr - 1); 
%     center = [cos(rot_angle) -sin(rot_angle); sin(rot_angle), cos(rot_angle)]*center;
%     centerAll{nr} = center;
%     
% end

% % % 3 centers each set
for nr = 1:max(size(D))

    % generate the initial set of 3 centers for targets.
%     center = repmat([0, -9]', [1 3]) + D(nr)*[-1, 0, 1; (2-sqrt(3)), 2, 2-sqrt(3)];
    center = repmat([0, -R/2]', [1 3]) - D(nr)*[-1, 1, 0; (2-sqrt(3)),  2-sqrt(3),2];
%     center = [0,-3,3; -2.3038,-7.5,-7.5];
    % rotate pi/3 degrees to generate more targets of different sizes
    rot_angle = pi/3*(nr - 1); 
    center = [cos(rot_angle) -sin(rot_angle); sin(rot_angle), cos(rot_angle)]*center;
    centerAll{nr} = center;
    
end

% % 6 centers each set
% for nr = 1:max(size(D))
% 
%     % generate the initial set of 6 centers for targets.
% 
%     center = repmat([0, -R/2]', [1 6]) + D(nr)*[0 -1 1 -2 0 2; sqrt(3)*[1 0 0 -1 -1 -1]];
% %     ete = 1.5;
% %     center = repmat([0, -R/2]', [1 6]) + (D(nr)+ete)*[0.5*[0 -1 1 -2 0 2]; (sqrt(3)/2)*[1 0 0 -1 -1 -1]];
% %     center = [0, -0.6, 0.6,-1.2, 0, 1.2; -2, -3.2, -3.2, -4.4, -4.4, -4.4 ];
% %     center = [0, -0.5, 0.5 , -1 , 0 , 1;-2.2,-3.2,-3.2,-4.2,-4.2,-4.2];
% 
%     
%     % rotate pi/3 degrees to generate more targets of different sizes
%     rot_angle = pi/3*(nr - 1); 
%     centeri = [cos(rot_angle) -sin(rot_angle); sin(rot_angle), cos(rot_angle)]*center;%旋转对象是物体，每旋转一个角度，相应的改变目标体的坐标*
%     centerAll{nr} = centeri;
%     
% end

% simulate target positions, which are within the small circles.
target_pixels =  false(size(xi));
for ii = 1:max(size(D))
    
    centeri = centerAll{ii};
    
    for jj = 1:size(centeri,2)
    
        center = centeri(:,jj);
        nodes_5mm = ( (xi-center(1)).^2 + (yi-center(2)).^2 <= 1/4*D(ii)^2);
        if D_inner(ii) > 0
            nodes_5mm = ( (xi-center(1)).^2 + (yi-center(2)).^2 <= 1/4*D(ii)^2) & ((xi-center(1)).^2 + (yi-center(2)).^2 >= 1/4*D_inner(ii)^2);
        end
        target_pixels = target_pixels|nodes_5mm ;
        
        
    end
    
end
target_pixels = reshape(target_pixels, size(xi));
% save center.mat;

%% Get the interpolation matrix of P
for i=1
change_flag = 0;
P_name = ['newFIBERP_R',num2str(R),'_H',num2str(height),'_Pix',num2str(pix_size),'4_Fibers','.mat'];
% GreenDx_name = ['newFIBERP_R',num2str(R),'_H',num2str(height),'_Pix',num2str(pix_size),'4_Fibers','.mat'];
if exist(P_name,'file')
    load(P_name);
    else

        switch change_flag
            case 0
                if exist('P_matrix.mat','file')
                    load P_matrix;
                else % Generate P

                    % load previously generated cylindrical phantom mesh
                    elements = load('mice_vols.txt'); elements = elements(:,2:5);
                    nodes = load('mice_nodes.txt');
                    faces = load('mice_faces.txt'); faces = faces(:,2:4);

                    % scale nodes to new phantom
                    x = nodes(:,1);   x = x/max(abs(x)) * R; 
                    y = nodes(:,2);   y = y/max(abs(y)) * R; 
                    z = nodes(:,3);   z = z/max(abs(z)) * height;       
                    nodes = [x y z]; 

                    % set detectors to be nodes on top surface
        %             detectors = find(z == height & sqrt(x.^2 + y.^2) < 0.95*R);


        %--- Set detectors as fiber bundles surrounding phantom ----%
                    N_nodes = size(nodes,1);
                    %detector = zeros(4,3);
                    detectors = zeros(16,1); % (#detectors, 1)
                    for i=1:16 %1, 6 % # detectors 
                        % detector = [ 0 0 0 ]; % [x y z] (middle of top surface (0,0,0))
                        detector = [ R*cos(2*pi/16*(i-1)) R*sin(2*pi/16*(i-1)) depth-2 ]; % set detectors 2 mm below scan section
                        sub(:,1) = nodes(:,1) - detector(1);
                        sub(:,2) = nodes(:,2) - detector(2);
                        sub(:,3) = nodes(:,3) - detector(3);
                        dista = zeros(N_nodes,1);
                        for j=1:N_nodes
                            dista(j) = norm(sub(j,:),2);% p-norm,p=2
                        end
                        [C,I] = min(dista);
                        detectors(i,1) = I;
                    end


                    amu = 0.0072; smu = 0.72; alfa = 0.128;

                    [P] = FOT_forward_P(nodes,elements,faces,detectors,amu,smu,alfa);%P为格林函数
                    save P_matrix P nodes;
                end
                
            case 1
                if exist('GreenDx.mat','file')
                    load GreenDx;
                    P = GreenDx;
                else % Generate P
                    
                    % load previously generated cylindrical phantom mesh
% %                     elements = load(sprintf('cylinder_elements_op.dat')); elements = elements(:,2:5);
% %                     nodes = load(sprintf('cylinder_nodes_op.dat')); nodes = nodes(:,2:4);
% %                     faces = load(sprintf('cylinder_faces_op.dat')); faces = faces(:,2:4);

                    elements = load('mice_vols.txt'); elements = elements(:,2:5);
                    nodes = load('mice_nodes.txt');
                    faces = load('mice_faces.txt'); faces = faces(:,2:4);

                    % scale nodes to new phantom
                    x = nodes(:,1);   x = x/max(abs(x)) * R; 
                    y = nodes(:,2);   y = y/max(abs(y)) * R; 
                    z = nodes(:,3);   z = z/max(abs(z)) * height;       
                    nodes = [x y z]; 

                    amu = 0.0072; smu = 0.72; alfa = 0.128;
                    [GreenDx] = Make__Matrix_RBC_Detector(nodes,elements,faces,amu,smu,alfa,depth-2);
                    save GreenDx
                    P = GreenDx;
                end
          otherwise
               warning('Unexpected plot type. No plot created.')
        end

        
    % Interpolate P onto slice: z = height - depth
    disp('P is being interpolated ...')
    newP = zeros(size(xi,1)*size(xi,2),size(P,2));
    xi_vec = xi(:); yi_vec = yi(:); zi_vec = zi(:);
    parfor i=1:size(P,2)
        FF = scatteredInterpolant(nodes,P(:,i));
        newP(:,i) = max(FF(xi_vec, yi_vec, zi_vec),0);
    end
    P = newP;clear nexP;
%     save(['newFIBERP_R',num2str(R),'_H',num2str(height),'_Pix',num2str(pix_size),'4_Fibers','.mat'], 'P');
    disp('Finished interpolation of P.')
    delete(gcp);
end
end
%% Nested Iterations. First, generate excitation matrix F.

% pix_per_beam = ceil(beam_width/pix_size);%每个光束所占的节点的数量或者是像素
length = size(xi,2);

for nScan = 0; % how many scans
    
    % Generate a circular mask that extract the circle out of a squared region
    mask = ones(length, length);
%     mask = ones(512, 512);
    scan_radius = floor(length/2);
%     scan_radius = 256;
    [hori, verti] = meshgrid(1:length, 1:length);
    out_circle = ((hori - scan_radius).^2 + (verti - scan_radius).^2 > scan_radius.^2);
    mask(out_circle) = 0;

    mask2 = nan(length, length);
    mask2(~out_circle) = 1;
    

    
for nProj = nProj_all; % how many projections?

%     % Initialize reconstruction nodes for all projections and scans
    reconNodes = zeros(nProj, length, length);
   
for theta0 = F_pert_angle_all % if there's perturbation for F
    
%     load('mask_new.mat')
%    load('x_mask_test5_10_60pos_5masks.mat'); 
  load('x_mask_test2_10_30pos.mat'); %mask_element = 20; pix_size_rev = 0.3;
%     load('x_mask_test5_10_60pos.mat'); 
%    x_mask = x_mask1;
    % Get the excitation matrix
    % Scans start from a horizontal line from top to bottom, then rotate
      for iProj = 1:nProj
          for j = 1: nmask
              if iProj >=2
                  for m = 1:3
                      select = m;
                      switch select
                            case 1
                                x_mask = x_mask1;
                            case 2
                                x_mask = x_mask2;
                            otherwise
                                 x_mask = x_mask3;
                      end
                  end
              else
                  x_mask = x_mask1;
                  dataR = imrotate(x_mask, -360/nProj*(iProj-1) - theta0,'crop');
                  reconNodes(nmask*(iProj-1)+j,:,:) = dataR.*mask;
                          
% 
%                 select = iProj-3*floor(iProj/3);
%                 select = round(unifrnd(0,1,1,1)*2);
%                 select = j;
%                 switch select
%                     case 1
%                         x_mask = x_mask1;
%                     case 2
%                         x_mask = x_mask2;
% 
% %                     case 3 
% %                         x_mask = x_mask3;
% %                     case 4
% %                         x_mask = x_mask4;
% %                     case 5
% %                         x_mask = x_mask1;
% %                     case 6
% %                         x_mask = x_mask4;
% %                     case 7
% %                         x_mask = x_mask3;
% %                     case 8
% %                         x_mask = x_mask2;
% %                     case 9
% %                         x_mask = x_mask1;
% % %                     case 10
% % %                         x_mask = x_mask2;
% % %                     case 11
% % %                         x_mask = x_mask5;
%                     otherwise
%                         x_mask = x_mask3;
%                 end
                
                      
            dataR = imrotate(x_mask, -360/nProj*(iProj-1) - theta0,'crop');
%             nProj = 60; iProj = 1;  theta0 = 0; 
%             dataR18 = imrotate(x_mask3, -102,'crop');
% %             data = dataR1|dataR2|dataR3|dataR4|dataR5|dataR6;%|dataR7|dataR8|dataR9
%             data = dataR1|dataR2|dataR3|dataR4|dataR5|dataR6|dataR7|dataR8|dataR9|dataR10|dataR11|dataR12...
%                 |dataR13|dataR14|dataR15|dataR16|dataR17|dataR18;%|dataR19;
%             perc = sum(sum(data))/(600*600)
%             perc = sum(sum(data))/(3.14*15^2/(0.05^2))
% %             count = 0;
% %             for i=1:256
% %                  Sum = sum(data(:,i));
% %                  if Sum < 64
% %                      n = i;
% %                      count = count+1;
% %                  end
% %             end           
%              test_mask = imrotate(test, + 102,'crop');
%              [x,y] = find(test_mask==1);

            reconNodes(nmask*(iProj-1)+j,:,:) = dataR.*mask;  
 %--------------------------------  perc  --------------------------------- 
%             num_all  = zeros(1,300,300);
%              for i = 1:72
%                  num_all = num_all|reconNodes(i,:,:);
%              end
%              perc_all = sum(num_all(:))/(round(3.14*15^2/(0.1^2)))
%              perc_all = size(find(num_all==1),1)/(round(3.14*6.4^2/(0.05^2)))
 %-------------------------------------------------------------------------      

         end

     end

    xrayNodes = reshape(reconNodes(1:nmask*nProj,:,:), [nmask*nProj,(length)^2]);

    F = xrayNodes'; clear xrayNodes;

%% Simulate measurement

for noise_level = noise_level_all
    
    Yname = ['Y_DerenzoD',num2str(D), '_nProj', num2str(nProj), '_perturb',num2str(theta0),'_noise',num2str(noise_level),'p.mat'];
    if exist(Yname,'file');
        load(Yname);
    else
        x_real = zeros(length, length); 
        x_real(logical(target_pixels))=1; 
        Y = P' * ( F .* repmat(x_real(:),[1,size(F,2)]) ); 
        Ynoise = Y.*(ones(size(Y)) + noise_level_all*(rand(size(Y))-0.5)); % add white noise
%         save(Yname,'Ynoise', 'x_real','Y');
    end

%% Reduce P and F by applying XCT mask
if XCT_mask;
    
    max_Y = max(Ynoise(:));
    n2k = ones(size(F,1),1);
    source2k = [];
%     nScan = nScan_per_beam;
    
    for i = 1:nProj
       
        % Nodes to keep for each proj
        n2kj = zeros(size(F,1),1);
       
        % Search each scan for nodes to keep 
%         for j = 1:nScan
            
            col_j = i;
             max_j = max(Ynoise(:, col_j));
            if  max_j > threshold_percentage * max_Y;
                
                n2kj = n2kj|F(:,col_j);
                source2k = [source2k; col_j];

            end
%         end
        
        % Intersection from all projections
        n2k = n2k & n2kj;
        
        disp(['nodes kept after ', num2str(i), ' projections: ',num2str(sum(n2k))]);

    end

    F2k = F(n2k,:); P2k = P(n2k,:);  Y2k = Ynoise; 
    
%     % Use less measurements    
%     F2k = F2k(:,source2k); Y2k = Ynoise(:,source2k); 

else
    
    F2k = F; P2k = P; Y2k = Ynoise;

end

%{
%%%%%%%%%%%%%%%%%% PCG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
almda=5.0d+6;
iter = 10;
%Initial image
% x0=ones(nScan*nScan,1)*0.00001;
x0=ones(sum(n2k),1)*0.00001;
tic;
[x,xs,cost]=FOT_Rec_Tikhonov_PCG(permute(P2k,[2,1,3]),F2k,Y,x0,iter,almda);
toc;
v=xs(end,:);
%}

%% %%%%%%%%%%%%%%%% NUMOS with Nesterov %%%%%%%%%%%%%%%%%%%
%逆向重建--二维格林函数
%     % Interpolate P onto slice: z = height - depth
%     [xi,yi,zi] = meshgrid(-R:pix_size_rev:(R-pix_size_rev/2), -R:pix_size_rev:(R-pix_size_rev/2), height-depth);
%     disp('P is being interpolated ...')
%     load P_matrix.mat;
%     newP = zeros(size(xi,1)*size(xi,2),size(P,2));
%     xi_vec = xi(:); yi_vec = yi(:); zi_vec = zi(:);
%     parfor i=1:size(P,2)
%         FF = scatteredInterpolant(nodes,P(:,i));
%         newP(:,i) = max(FF(xi_vec, yi_vec, zi_vec),0);
%     end
%     P = newP;clear newP;
% %     save(['newFIBERP_R',num2str(R),'_H',num2str(height),'_Pix',num2str(pix_size),'4_Fibers','.mat'], 'P');
%     disp('Finished interpolation of P.')
%     delete(gcp);
% %激发矩阵F_re
% length = size(xi,2);
% for nScan1 = 0; % how many scans
%     
%     % Generate a circular mask that extract the circle out of a squared region
%     mask = ones(length, length);
% %     mask = ones(512, 512);
%     scan_radius = floor(length/2);
% %     scan_radius = 256;
%     [hori, verti] = meshgrid(1:length, 1:length);
%     out_circle = ((hori - scan_radius).^2 + (verti - scan_radius).^2 > scan_radius.^2);
%     mask(out_circle) = 0;
% 
%     mask2 = nan(length, length);
%     mask2(~out_circle) = 1;
%       
% for nProj = nProj_all; % how many projections?
% 
% %     % Initialize reconstruction nodes for all projections and scans
%     reconNodes = zeros(nProj, length, length);
%     load('20element_10pins.mat');
%     for iProj = 1:nProj
%           for j = 1: nmask              
%             select = iProj-3*floor(iProj/3);
% %                 select = j;
%             switch select
%                 case 1
%                     x_mask = x_mask1;
%                 case 2
%                     x_mask = x_mask2;
% %                     case 3 
% %                         x_mask = x_mask3;
% %                     case 4
% %                         x_mask = x_mask4;
%                 otherwise
%                     x_mask = x_mask3;
%             end
%             dataR = imrotate(x_mask, -360/nProj*(iProj-1) - theta0,'crop');
%             reconNodes(nmask*(iProj-1)+j,:,:) = dataR.*mask;  
%           end
%      end
% 
%     xrayNodes = reshape(reconNodes(1:nmask*nProj,:,:), [nmask*nProj,(length)^2]);
% 
%     F_rev = xrayNodes'; clear xrayNodes;
% end
% end
%     P2k = P; F2k = F_rev;
%-----------------------------------------------------------------------%
for ip = percentage
    
    % regularization parameter  
    max_Aty = max(sum(F2k.*(P2k * Y2k),2)); 
    lambda1 = ip/300*max_Aty; nblock = 1;fast = 2; AorM = 'M'; tol = []; delta = []; 
    
    x0 = ones(size(P2k,1),1)*0.00001; % uniform initialization
    output = MM_recon(F2k,P2k',Y2k, lambda1, IterNumMax, x0, tol, nblock, fast, AorM, 1, delta);
   
    %% Plotting
    
%     for iiT=10:10:IterNumMax
    for iiT = IterNumMax:IterNumMax
        v = output.x(:,iiT);
        if size(v) < numel(xi);    
%             v = 0.1*ones(numel(xi),1);  
            v = zeros(numel(xi),1); 
            v(n2k) = output.x(:,iiT); %    
        end
        
        img = reshape(v,size(xi));
        img(isnan(img))=0;
        img=filter2(fspecial('average',7),img);
        hFig1 = figure();
        set(hFig1,'Position', [100, 100, 650, 512]);

        surf(xi,yi,zi,img.*mask2); shading flat;grid off;
   
        hold on;
        
for nr=1:max(size(D))

    center = centerAll{nr};
    
    for nc=1:size(center,2)
        THETA=linspace(0,2*pi,1000);
        RHO=ones(1,1000) * D(nr)/2;
        [xx,yy] = pol2cart(THETA,RHO);
        xx = xx + center(1,nc)+1/2*pix_size;
        yy = yy + center(2,nc)+1/2*pix_size;
        zz = (height -depth)*ones(size(yy));
        H=plot3(xx,yy,zz,'r-');
        
        % Draw inner ring
        if D_inner(nr) > 0; 
            RHO = ones(1,1000)*D_inner(nr)/2;
            [xx,yy] = pol2cart(THETA,RHO);
            xx = xx + center(1,nc)+1/2*pix_size;
            yy = yy + center(2,nc)+1/2*pix_size;
            zz = (height -depth)*ones(size(yy));
            H=plot3(xx,yy,zz,'r-');
        end
    end

end

        set(gca,'FontSize',20,'FontName','Times New Roman');
        hcb = colorbar('FontSize',20,'FontName','Times New Roman');
        colorTitleHandle = get(hcb,'Title');
        colormap('JET');
        titleString = '[a.u.]';
        set(colorTitleHandle ,'String',titleString);
        xlabel('x (mm)','FontSize',20,'FontName','Times New Roman');
        ylabel('y (mm)','FontSize',20,'FontName','Times New Roman');
               
        axis equal; view(0,90);
        
%         axis([-4 4 -12 -4]);   
%         axis([-6.5 6.5 -6.5 6.5]);
        axis([-15 15 -15 15]);        
 
    end
end

end
end
end
end




% % % %% Dice

Rec = img;
MaxI = max(img(:));
R1 = D(1)/2;
cir1= ( (xi -centeri(1,1)).^2 + (yi - centeri(2,1)).^2 <= R1.^2); % first target circle
cir2= ( (xi - centeri(1,2)).^2 + (yi - centeri(2,2)).^2 <= R1.^2); % second target circle
cir3= ( (xi - centeri(1,3)).^2 + (yi - centeri(2,3)).^2 <= R1.^2); % first target circle
% % cir4= ( (xi - centeri(1,4)).^2 + (yi -centeri(2,4)).^2 <= R1.^2); % second target circle
% % cir5= ( (xi -centeri(1,5)).^2 + (yi -centeri(2,5)).^2 <= R1.^2); % first target circle
% % cir6= ( (xi -centeri(1,6)).^2 + (yi -centeri(2,6)).^2 <= R1.^2); % second target circle
% 
% % cir = or(cir1,cir2,cir3,cir4,cir5,cir6); % either in cir1 or cir2 (thus use or)
cir = cir1|cir2|cir3;
AS = sum(cir(:));
rec = (Rec>0.1*MaxI);
RS = sum(rec(:));
rec1 = and(cir,rec);
AS_RS = sum(rec1(:));
DICE = 2*AS_RS/(AS+RS)*100 % DICE Coefficient
% 
% %%-------------------------------------------------------------------------%%
% %%% 目标体尺寸误差Target size error（TSE）3个目标体取平均值(location error ,LE)
half_h = (center(2,1)-center(2,3))/2;
x1 = (center(1,1)-D(1)):pix_size:(center(1,1)+D(1));y1 = (center(2,1)-D(1)):pix_size:(center(2,1)+D(1));
col_x = round((R+x1)/pix_size);row_y = round((R+y1)/pix_size);
MaxVal = max(max(Rec(row_y,col_x))); 
[row ,col] = find(Rec==MaxVal);
x11 = col*pix_size-R;
y11 = row*pix_size-R;
center11 = [x11 y11]';
LE1 = norm(center11-center(:,1));
% Dr1 = sum(Rec(row_y,col)>=0.5*MaxVal)*0.1;
% Dr2 = sum(Rec(row,col_x)>=0.5*MaxVal)*0.1;
% Dr = (Dr1+Dr2)/2;
% Dt = D(1);
% TSE1 = abs(Dt-Dr)/Dt;
% 
x2 = (center(1,2)-D(1)):pix_size:(center(1,2)+D(1));y2 = (center(2,2)-D(1)):pix_size:(center(2,2)+D(1));
col_x = round((R+x2)/pix_size);row_y = round((R+y2)/pix_size);
MaxVal = max(max(Rec(row_y,col_x))); 
[row ,col] = find(Rec==MaxVal);
x11 = col*pix_size-R;
y11 = row*pix_size-R;
center11 = [x11 y11]';
LE2 = norm(center11-center(:,2));
% Dr1 = sum(Rec(row_y,col)>=0.5*MaxVal)*0.1;
% Dr2 = sum(Rec(row,col_x)>=0.5*MaxVal)*0.1;
% Dr = (Dr1+Dr2)/2;
% Dt = D(1);
% TSE2 = abs(Dt-Dr)/Dt;
% 
x3 = (center(1,3)-D(1)):pix_size:(center(1,3)+D(1));y3 = (center(2,3)-D(1)):pix_size:(center(2,3)+D(1));
col_x = round((R+x3)/pix_size);row_y = round((R+y3)/pix_size);
MaxVal = max(max(Rec(row_y,col_x))); 
[row ,col] = find(Rec==MaxVal);
x11 = col*pix_size-R;
y11 = row*pix_size-R;
center11 = [x11 y11]';
LE3 = norm(center11-center(:,3));

LE = (LE1+LE2+LE3)/3
% Dr1 = sum(Rec(row_y,col)>=0.5*MaxVal)*0.1;
% Dr2 = sum(Rec(row,col_x)>=0.5*MaxVal)*0.1;
% Dr = (Dr1+Dr2)/2;
% Dt = D(1);
% % TSE3 = abs(Dt-Dr)/Dt;
% TSE = (TSE1+TSE2+TSE3)/3*100

%  %%% 标准均方根误差 NMSE(mean sqart error)
rho_t = x_real;%the true intensity
rho_r = Rec;
NMSE = (norm(rho_t-rho_r))^2/(norm(rho_t))^2
  
% % %%% 空间分辨率指数（spatial resolution index）SRI
% a = [sqrt(3)   -sqrt(3)   0];b = [ center(2,1)  center(2,2) center(2,3) ];
% ROI_min = 0;
% length = size(((center(1,1)-D(1)):0.1:(center(1,2)+D(1))),2);
% ROI_max = zeros(1,3);
% % % x_spi = zeros(2,:);
% length = size(((center(1,1)-D(1)):0.1:(center(1,2)+D(1))),2);
% x_spi = zeros(2,length);
% x_spi(1,:) = center(1,2):0.025:center(1,6);
% x_spi(2,:)= center(1,4):0.025:center(1,3);
% for i =1:2
%     y = a(i)*x_spi(i,:)+b(i);
%     col_x = round((6.4+x_spi(i,:))/0.025);
%     row_y = round((6.4+y)/0.025);
%     ROI_max(i) = max(max(Rec(row_y,col_x)));
% end
%     x3 = center(1,4):0.025:center(1,6);
%     y = a(3)*x3+b(3);
%     col_x = round((6.4+x3)/0.025);
%     row_y = round((6.4+y)/0.025);
%     ROI_max(3) = max(max(Rec(row_y,col_x)));
%     
% ROI_valley = zeros(1,3);
% length1 = size((center(1,5):0.025:center(1,3)),2);
% x_spi1 = zeros(2,length1);
% x_spi1(1,:) = center(1,5):0.025:center(1,3);
% x_spi1(2,:) = center(1,2):0.025:center(1,5);
% for j = 1:2
%     y1 = a(j)*x_spi1(j,:)+b(j);
%     col_x1 = round((6.4+x_spi1(j,:))/0.025);
%     row_y1 = round((6.4+y1)/0.025);
%     ROI_valley(j) = min(min(Rec(row_y1,col_x1)));
% end
% 
%     x13 = center(1,2):0.025:center(1,3);
%     y1 = a(3)*x13+b(3);
%     col_x1 = round((6.4+x13)/0.025);
%     row_y1 = round((6.4+y1)/0.025);
%     ROI_valley(3) = min(min(Rec(row_y1,col_x1)));
% 
% SPI = (ROI_max - ROI_valley)./(ROI_max - ROI_min);
% SPI = mean(SPI)


