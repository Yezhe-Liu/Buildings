clear all;
close all;

%dir="D:\local\pathloss250808\li\chizhou140zgg\pred";
dir="D:/ISAC/PlotSource/pred";
       Color = [-115 -60];

    indexfile = sprintf('%s\\index.txt',dir);
    demfile = sprintf('%s\\RSRP.dem',dir);

    [left1,right1,bottom1,top1,res1] = textread(indexfile, '%f %f %f %f %f');
    M1=floor((top1-bottom1)/res1);
    N1=floor((right1-left1)/res1);

    fid = fopen(demfile,'rb');
    RSRPdem1=zeros(M1,N1);
    Cord=[];
    Cord = [left1 right1 bottom1 top1 res1];
    m = 0;
    while feof(fid) == 0
        m = m + 1;
        [row_array,ele_count]=fread(fid,N1,'int16');
        if ele_count<N1
            break;
        else
            row_array=row_array';
            %PL=[PL; row_array];
            RSRPdem1(m,:) = row_array;
        end
    end
    fclose(fid);

    RSRPdem1=RSRPdem1/100;
    alphadata = ones(M1,N1) * 0.7;
    alphadata(find(RSRPdem1==-327.68)) = 0;%设置透明度
    
    X = 0:Cord(5):Cord(2)-Cord(1);
    Y = Cord(4)-Cord(3):-Cord(5):0;
    %Y = 0:Cord(5):Cord(4)-Cord(3);
    im = imagesc(X,Y,RSRPdem1,Color);
    set(gca,'YDir','NORMAL');
    im.AlphaData = alphadata;
    colormap(flipud(jet))
    colorbar;

