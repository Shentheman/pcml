#include <pcml/data/cad120_reader.h>

#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <sensor_msgs/PointCloud2.h>

namespace pcml
{

bool CAD120Reader::existDirectory(const std::string& directory)
{
    struct stat sb;
    return lstat(directory.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}


CAD120Reader::CAD120Reader()
    : CAD120Reader("~")
{
}

CAD120Reader::CAD120Reader(const std::string& directory)
{
    rgbd_fp_ = NULL;

    setDirectory(directory);
}


void CAD120Reader::setDirectory(const std::string& directory)
{
    if (directory == "")
        directory_ = "~";

    else if (*directory.rbegin() == '/')
        directory_ = directory.substr(0, directory.size() - 1);

    else
        directory_ = directory;

    if (!existDirectory(directory_))
    {
        fprintf(stderr, "Invalid CAD120 dataset directory [%s]", directory_.c_str());
        fflush(stderr);
        return;
    }

    for (int i=1; i<=4; i++)
    {
        char annotation_directory[NAME_MAX + 1];
        sprintf(annotation_directory, "%s/Subject%d_annotations", directory_.c_str(), i);
        if (existDirectory(annotation_directory))
        {
            Subject subject;
            subject.subject_id = i;

            std::vector<Action> actions;
            struct dirent* de = NULL;
            DIR* d = opendir(annotation_directory);
            while (de = readdir(d))
            {
                std::string action_id = de->d_name;
                if (action_id == "." || action_id == "..")
                    continue;

                char action_directory[NAME_MAX + 1];
                sprintf(action_directory, "%s/%s", annotation_directory, action_id.c_str());

                Action action;
                action.action_id = action_id;
                readActionInfo(action, action_directory);
                actions.push_back(action);
            }
            closedir(d);

            subject.actions = actions;
            subjects_.push_back(subject);
        }
    }
}

void CAD120Reader::readActionInfo(Action& action, const std::string& action_directory)
{
    char filename[NAME_MAX + 1];
    sprintf(filename, "%s/activityLabel.txt", action_directory.c_str());

    FILE* fp = fopen(filename, "r");

    char buffer[1024];
    while (fgets(buffer, 1024, fp))
    {
        Video video;

        char* p = strtok(buffer, ",");
        video.id = p;

        // global transform
        sprintf(filename, "%s/%s_globalTransform.txt", action_directory.c_str(), video.id.c_str());
        FILE* tf_fp = fopen(filename, "r");
        for (int i=0; i<4; i++)
        {
            for (int j=0; j<4; j++)
                fscanf(tf_fp, "%lf%*c", &video.global_transform(i,j));
        }
        fclose(tf_fp);

        p = strtok(NULL, ",");
        video.activity_id = p;

        p = strtok(NULL, ",");
        video.subject_id = p;

        while (true)
        {
            p = strtok(NULL, ",:\n");
            if (p == NULL) break;
            video.object_ids.push_back(atoi(p));

            p = strtok(NULL, ",:\n");
            video.object_types.push_back(p);
        }

        action.videos.push_back(video);
    }

    fclose(fp);
}

void CAD120Reader::print()
{
    printf("%d subjects\n", subjects_.size());
    for (int i=0; i<subjects_.size(); i++)
    {
        printf(" subject [%d]:\n", subjects_[i].subject_id);
        for (int j=0; j<subjects_[i].actions.size(); j++)
        {
            printf("  action [%s]:\n", subjects_[i].actions[j].action_id.c_str());
            for (int k=0; k<subjects_[i].actions[j].videos.size(); k++)
            {
                const Video& video = subjects_[i].actions[j].videos[k];
                printf("   video [%s]: ", video.id.c_str());
                printf("activity [%s], subject [%s], objects ", video.activity_id.c_str(), video.subject_id.c_str());
                for (int l=0; l<video.object_ids.size(); l++)
                {
                    printf("[%d:%s]", video.object_ids[l], video.object_types[l].c_str());
                    if (l != video.object_ids.size() - 1)
                        printf(", ");
                }
                printf("\n");
            }
        }
    }

    fflush(stdout);
}

int CAD120Reader::numSubjects()
{
    return subjects_.size();
}

int CAD120Reader::numActions(int subject)
{
    return subjects_[subject].actions.size();
}

int CAD120Reader::numVideos(int subject, int action)
{
    return subjects_[subject].actions[action].videos.size();
}

void CAD120Reader::startReadFrames(int subject, int action, int video)
{
    finishReadFrames();

    subject_ = subject;
    action_ = action;
    video_ = video;

    const Video& v = subjects_[subject].actions[action].videos[video];
    char filename[NAME_MAX + 1];
    sprintf(filename, "%s/Subject%d_rgbd_rawtext/%s/%s_rgbd.txt",
            directory_.c_str(), subjects_[subject].subject_id, subjects_[subject].actions[action].action_id.c_str(), v.id.c_str());

    rgbd_fp_ = fopen(filename, "r");
}

bool CAD120Reader::readNextFrame()
{
    // 6(number+comma) * 4(channels) * 640(X_RES) * 480(Y_RES) = 7372800 Bytes = 7 MB
    static const int buffer_size = 6 * 4 * X_RES * Y_RES;
    static char buffer[buffer_size];

    if (fgets(buffer, buffer_size, rgbd_fp_) == NULL)
        return false;

    // first integer is the frame number
    char* p = strtok(buffer, ",\n");
    int* ptr = (int*)rgbd_image_;

    while (true)
    {
        p = strtok(NULL, ",\n");
        if (p == NULL) break;

        *ptr = atoi(p);
        ptr++;
    }

    return true;
}

void CAD120Reader::finishReadFrames()
{
    if (rgbd_fp_ != NULL)
    {
        fclose(rgbd_fp_);
        rgbd_fp_ = NULL;
    }
}

void CAD120Reader::setPointCloudTopic(const std::string& topic)
{
    ros::NodeHandle n;

    point_cloud_publisher_ = n.advertise<sensor_msgs::PointCloud2>(topic, 100);
}

void CAD120Reader::renderPointCloud()
{
    const Eigen::Matrix4d& transform = subjects_[subject_].actions[action_].videos[video_].global_transform;

    sensor_msgs::PointCloud2 point_cloud;
    sensor_msgs::PointField field;
    char buffer[4];
    float* fptr = (float*)buffer;

    point_cloud.header.frame_id = "/world";
    point_cloud.header.stamp = ros::Time::now();

    point_cloud.height = Y_RES;
    point_cloud.width = X_RES;

    point_cloud.is_bigendian = false;
    point_cloud.is_dense = false;

    point_cloud.point_step = 16;
    point_cloud.row_step = 16 * X_RES;

    // point = [x(4) y(4) z(4) rgb(4)]
    field.count = 1;

    field.datatype = sensor_msgs::PointField::FLOAT32;
    field.name = "x";
    field.offset = 0;
    point_cloud.fields.push_back(field);
    field.name = "y";
    field.offset = 4;
    point_cloud.fields.push_back(field);
    field.name = "z";
    field.offset = 8;
    point_cloud.fields.push_back(field);

    field.datatype = sensor_msgs::PointField::UINT32;
    field.name = "rgb";
    field.offset = 12;
    point_cloud.fields.push_back(field);

    for (int i=0; i<Y_RES; i++)
    {
        for (int j=0; j<X_RES; j++)
        {
            /*
            cloud.points.at(index).x = (x - 640 * 0.5) * cloud.points.at(index).y * 1.1147 / 640;
            cloud.points.at(index).z = (480 * 0.5 - y) * cloud.points.at(index).y * 0.8336 / 480;
            */

            Eigen::Vector4d v;
            v(1) = rgbd_image_[i][j][3];
            v(0) = (j - X_RES * 0.5) * v(1) * 1.1147 / X_RES;
            v(2) = (Y_RES * 0.5 - i) * v(1) * 0.8336 / Y_RES;
            v(3) = 1;

            v = transform * v;
            v(0) /= v(3);
            v(1) /= v(3);
            v(2) /= v(3);
            v /= 1000;

            *fptr = v(0);
            for (int k=0; k<4; k++)
                point_cloud.data.push_back(buffer[k]);

            *fptr = v(1);
            for (int k=0; k<4; k++)
                point_cloud.data.push_back(buffer[k]);

            *fptr = v(2);
            for (int k=0; k<4; k++)
                point_cloud.data.push_back(buffer[k]);

            /*
            *fptr = (double)j / X_RES;
            for (int k=0; k<4; k++)
                point_cloud.data.push_back(buffer[k]);

            *fptr = (double)i / X_RES;
            for (int k=0; k<4; k++)
                point_cloud.data.push_back(buffer[k]);

            *fptr = (double)rgbd_image_[i][j][3] / 65535;
            for (int k=0; k<4; k++)
                point_cloud.data.push_back(buffer[k]);
                */

            point_cloud.data.push_back(rgbd_image_[i][j][2]);
            point_cloud.data.push_back(rgbd_image_[i][j][1]);
            point_cloud.data.push_back(rgbd_image_[i][j][0]);
            point_cloud.data.push_back(0);
        }
    }

    point_cloud_publisher_.publish(point_cloud);
}

}
