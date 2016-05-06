#include <pcml/data/cad120_reader.h>

#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

namespace pcml
{

const char* CAD120Reader::joint_names_[num_joints_] =
{
    "head",
    "neck",
    "torso",
    "left_shoulder",
    "left_elbow",
    "right_shoulder",
    "right_elbow",
    "left_hip",
    "left_knee",
    "right_hip",
    "right_knee",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
};

const int CAD120Reader::skeletons_[num_skeletons_][2] =
{
     0,  1,
     1,  2,
     2,  3,
     3,  4,
     2,  5,
     5,  6,
     2,  7,
     7,  8,
     2,  9,
     9, 10,
     4, 11,
     6, 12,
     8, 13,
    10, 14,
};

const std::map<std::string, int> CAD120Reader::sub_activity_to_index_map_ =
{
    {"reaching", 0},
    {"moving"  , 1},
    {"pouring" , 2},
    {"eating"  , 3},
    {"drinking", 4},
    {"opening" , 5},
    {"placing" , 6},
    {"closing" , 7},
    {"cleaning", 8},
    {"null"    , 9},
};


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
    : marker_array_publisher_initialized_(false)
{
    rgbd_fp_ = NULL;
    joint_fp_ = NULL;
    object_fps_.clear();

    setDirectory(directory);
}

bool CAD120Reader::getJointPosition(const std::string& joint_name, Eigen::Vector3d& position)
{
    for (int i=0; i<num_joints_; i++)
    {
        if (joint_name == joint_names_[i])
        {
            // transform
            const Eigen::Matrix4d& transform = subjects_[subject_].actions[action_].videos[video_].global_transform;

            Eigen::Vector4d v;
            v << joint_pos_[i], 1.0;
            std::swap(v(1), v(2));

            v = transform * v;
            v(0) /= v(3);
            v(1) /= v(3);
            v(2) /= v(3);

            position = v.block(0, 0, 3, 1) / 1000.;

            return true;
        }
    }

    return false;
}

std::string CAD120Reader::getSubActivity()
{
    // current implementation is inefficient
    const std::vector<FrameIntervalAnnotation>& labelings = subjects_[subject_].actions[action_].videos[video_].annotations;

    for (int i=0; i<labelings.size(); i++)
    {
        if (labelings[i].start_frame <= current_frame_ && current_frame_ <= labelings[i].end_frame)
            return labelings[i].sub_activity_id;
    }

    // if not found in labeling file, then return the last sub-activity
    return labelings.rbegin()->sub_activity_id;
}

int CAD120Reader::getSubActivityIndex()
{
    std::map<std::string, int>::const_iterator it = sub_activity_to_index_map_.find(getSubActivity());
    if (it == sub_activity_to_index_map_.end())
        return -1;

    return it->second;
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
    current_frame_ = 0;

    const Video& v = subjects_[subject].actions[action].videos[video];
    char filename[NAME_MAX + 1];

    // rgbd
    sprintf(filename, "%s/Subject%d_rgbd_rawtext/%s/%s_rgbd.txt",
            directory_.c_str(), subjects_[subject].subject_id, subjects_[subject].actions[action].action_id.c_str(), v.id.c_str());
    rgbd_fp_ = fopen(filename, "r");

    // joints
    sprintf(filename, "%s/Subject%d_annotations/%s/%s.txt",
            directory_.c_str(), subjects_[subject].subject_id, subjects_[subject].actions[action].action_id.c_str(), v.id.c_str());
    joint_fp_ = fopen(filename, "r");

    // object
    object_fps_.resize( v.object_ids.size() );
    object_bounding_boxes_.resize( v.object_ids.size() );
    object_transformations_.resize( v.object_ids.size() );
    for (int i=0; i<v.object_ids.size(); i++)
    {
        sprintf(filename, "%s/Subject%d_annotations/%s/%s_obj%d.txt",
                directory_.c_str(), subjects_[subject].subject_id, subjects_[subject].actions[action].action_id.c_str(), v.id.c_str(), v.object_ids[i]);
        object_fps_[i] = fopen(filename, "r");
    }

    // labeling of temporal segments
    readLabeling();
}

void CAD120Reader::readLabeling()
{
    Video& v = subjects_[subject_].actions[action_].videos[video_];
    char filename[NAME_MAX + 1];
    static const int buffer_size = 256;
    static char buffer[buffer_size];

    sprintf(filename, "%s/Subject%d_annotations/%s/labeling.txt",
            directory_.c_str(), subjects_[subject_].subject_id, subjects_[subject_].actions[action_].action_id.c_str());

    FILE* labeling_fp = fopen(filename, "r");

    while (true)
    {
        if (fgets(buffer, buffer_size, labeling_fp) == NULL || strcmp(buffer, "\n") == 0)
            break;

        // line format: video_id, start_frame, end_frame, sub_activity, affordance(1), affordance(2), ...
        char* p = strtok(buffer, ",\n");
        if (v.id == p)
        {
            FrameIntervalAnnotation labeling;

            p = strtok(NULL, ",\n");
            labeling.start_frame = atoi(p);

            p = strtok(NULL, ",\n");
            labeling.end_frame = atoi(p);

            p = strtok(NULL, ",\n");
            labeling.sub_activity_id = p;

            while (true)
            {
                p = strtok(NULL, ",\n");
                if (p == NULL) break;

                labeling.affordance_ids.push_back(p);
            }

            v.annotations.push_back(labeling);
        }
    }

    fclose(labeling_fp);
}

bool CAD120Reader::readNextFrame()
{
    current_frame_++;

    bool is_opened = false; // whether any file is opened

    // RGBD data
    // 6(number+comma) * 4(channels) * 640(X_RES) * 480(Y_RES) = 7372800 Bytes = 7 MB
    static const int buffer_size = 6 * 4 * X_RES * Y_RES;
    static char buffer[buffer_size];

    if (rgbd_fp_ != NULL)
    {
        is_opened = true;

        if (fgets(buffer, buffer_size, rgbd_fp_) == NULL || strcmp(buffer, "\n") == 0)
            return false;

        int* ptr = (int*)rgbd_image_;

        // first integer is the frame number
        char* p = strtok(buffer, ",\n");
        while (true)
        {
            p = strtok(NULL, ",\n");
            if (p == NULL) break;

            *ptr = atoi(p);
            ptr++;
        }
    }

    // joint data
    if (joint_fp_ != NULL)
    {
        is_opened = true;

        if (fgets(buffer, buffer_size, joint_fp_) == NULL || strcmp(buffer, "\n") == 0)
            return false;

        // first integer is the frame number
        char* p = strtok(buffer, ",\n");
        if (strcmp(p, "END") != 0)
        {
            for (int i=0; i<num_joints_; i++)
            {
                if (i < num_joints_with_ori_)
                {
                    for (int j=0; j<3; j++)
                    {
                        for (int k=0; k<3; k++)
                        {
                            p = strtok(NULL, ",\n");
                            joint_ori_[i](j,k) = atof(p);
                        }
                    }
                    p = strtok(NULL, ",\n");
                    joint_ori_confidence_[i] = atoi(p);
                }

                for (int j=0; j<3; j++)
                {
                    p = strtok(NULL, ",\n");
                    joint_pos_[i](j) = atof(p);
                }
                p = strtok(NULL, ",\n");
                joint_pos_confidence_[i] = atoi(p);
            }
        }
    }


    // object data
    const Video& v = subjects_[subject_].actions[action_].videos[video_];
    for (int i=0; i<v.object_ids.size(); i++)
    {
        if (object_fps_[i] != NULL)
        {
            is_opened = true;

            if (fgets(buffer, buffer_size, object_fps_[i]) == NULL || strcmp(buffer, "\n") == 0)
                return false;

            // first two integer is the frame number and the object id
            char* p = strtok(buffer, ",\n");
            p = strtok(NULL, ",\n");

            // bounding box
            for (int j=0; j<4; j++)
            {
                p = strtok(NULL, ",\n");
                object_bounding_boxes_[i](j) = atoi(p);
            }

            // transformation
            for (int j=0; j<2; j++)
            {
                for (int k=0; k<3; k++)
                {
                    p = strtok(NULL, ",\n");
                    object_transformations_[i](j,k) = atof(p);
                }
            }
        }
    }

    return is_opened;
}

void CAD120Reader::finishReadFrames()
{
    if (rgbd_fp_ != NULL)
    {
        fclose(rgbd_fp_);
        rgbd_fp_ = NULL;
    }

    if (joint_fp_ != NULL)
    {
        fclose(joint_fp_);
        joint_fp_ = NULL;
    }

    for (int i=0; i<object_fps_.size(); i++)
    {
        if (object_fps_[i] != NULL)
            fclose(object_fps_[i]);
    }
    object_fps_.clear();


    // rviz object marker cleanup
    if (marker_array_publisher_initialized_)
    {
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker;
        marker.header.frame_id = "/world";
        marker.header.stamp = ros::Time::now();
        marker.ns = "objects";
        marker.action = visualization_msgs::Marker::DELETE;
        for (int i=0; i<object_bounding_boxes_.size() * 2; i++) // 0 ~ (n-1) for line list, n ~ (2n-1) for text
        {
            marker.id = i;
            marker_array.markers.push_back(marker);
        }
        marker_array_publisher_.publish(marker_array);
    }
}

void CAD120Reader::setPointCloudTopic(const std::string& topic)
{
    ros::NodeHandle n;
    point_cloud_publisher_ = n.advertise<sensor_msgs::PointCloud2>(topic, 100);
}

void CAD120Reader::setMarkerArrayTopic(const std::string& topic)
{
    ros::NodeHandle n;
    marker_array_publisher_ = n.advertise<visualization_msgs::MarkerArray>(topic, 100);
    marker_array_publisher_initialized_ = true;
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
    point_cloud.is_dense = true;

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
            /* in the downloaded code
            cloud.points.at(index).y = IMAGE[x][y][3];
            cloud.points.at(index).x = (x - 640 * 0.5) * cloud.points.at(index).y * 1.1147 / 640;
            cloud.points.at(index).z = (480 * 0.5 - y) * cloud.points.at(index).y * 0.8336 / 480;
            globalTransform.transformPointCloudInPlaceAndSetOrigin(cloud);
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

            point_cloud.data.push_back(rgbd_image_[i][j][2]);
            point_cloud.data.push_back(rgbd_image_[i][j][1]);
            point_cloud.data.push_back(rgbd_image_[i][j][0]);
            point_cloud.data.push_back(0);
        }
    }

    point_cloud_publisher_.publish(point_cloud);
}

void CAD120Reader::renderSkeleton()
{
    const Eigen::Matrix4d& transform = subjects_[subject_].actions[action_].videos[video_].global_transform;

    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;

    marker.header.frame_id = "/world";
    marker.header.stamp = ros::Time::now();

    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::LINE_LIST;

    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.w = 1.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;

    marker.ns = "skeleton";

    marker.scale.x = 0.03; // 3 cm

    marker.color.r = 1.;
    marker.color.g = 1.;
    marker.color.b = 0.;
    marker.color.a = 1.;

    for (int i=0; i<num_skeletons_; i++)
    {
        marker.id = i;

        for (int j=0; j<2; j++)
        {
            // The transformation is applied here

            /* in the downloaded code
            for (size_t i = 0; i < jointList.size(); i++) {
                    //pcl::PointXYZ p1 (data[i][9],data[i][11],data[i][10]);
                    pcl::PointXYZ pt ;
                    pt.x = data[jointList.at(i)][9];
                    pt.y = data[jointList.at(i)][11];
                    pt.z = data[jointList.at(i)][10];

                    globalTransform.transformPointInPlace(pt);
                    transformed_joints.push_back(pt);
            }
            for (size_t i = 0; i < pos_jointList.size(); i++) {
                    //pcl::PointXYZ p1 (data[i][9],data[i][11],data[i][10]);
                    pcl::PointXYZ pt ;
                    pt.x = pos_data[pos_jointList.at(i)][0];
                    pt.y = pos_data[pos_jointList.at(i)][2];
                    pt.z = pos_data[pos_jointList.at(i)][1];

                    globalTransform.transformPointInPlace(pt);
                    transformed_joints.push_back(pt);
            }
            */

            const int joint_id = skeletons_[i][j];
            Eigen::Vector4d v;
            v << joint_pos_[joint_id], 1.0;
            std::swap(v(1), v(2));

            v = transform * v;
            v(0) /= v(3);
            v(1) /= v(3);
            v(2) /= v(3);
            v /= 1000;

            geometry_msgs::Point point;
            point.x = v(0);
            point.y = v(1);
            point.z = v(2);
            marker.points.push_back(point);
        }
    }

    marker_array.markers.push_back(marker);

    marker_array_publisher_.publish(marker_array);
}

void CAD120Reader::renderObjects()
{
    const Eigen::Matrix4d& transform = subjects_[subject_].actions[action_].videos[video_].global_transform;

    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;

    marker.header.frame_id = "/world";
    marker.header.stamp = ros::Time::now();

    marker.ns = "objects";

    marker.pose.orientation.w = 1.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;

    marker.scale.x = 0.01; // line width = 1 cm
    marker.scale.z = 0.1;  // text height = 10 cm

    marker.color.a = 1.;

    for (int i=0; i<object_bounding_boxes_.size(); i++)
    {
        marker.id = i;

        std::vector<double> percentages;
        percentages.push_back(0.25);
        percentages.push_back(0.75);
        const std::vector<int> depths = findDepths( object_bounding_boxes_[i], percentages );

        // if bounding box is empty
        if (depths.empty())
        {
            marker.action = visualization_msgs::Marker::DELETE;
            marker_array.markers.push_back(marker);

            marker.id += object_bounding_boxes_.size();
            marker.action = visualization_msgs::Marker::DELETE;
            marker_array.markers.push_back(marker);
        }

        // if bounding box is defined
        else
        {
            // cube
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::LINE_LIST;

            marker.pose.position.x = 0.;
            marker.pose.position.y = 0.;
            marker.pose.position.z = 0.;
            marker.color.r = 1.;
            marker.color.g = 0.;
            marker.color.b = 0.;

            std::vector<Eigen::Vector3d> cube;
            for (int j=0; j<2; j++)
            {
                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](0), object_bounding_boxes_[i](1), depths[j] ) );
                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](2), object_bounding_boxes_[i](1), depths[j] ) );

                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](2), object_bounding_boxes_[i](1), depths[j] ) );
                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](2), object_bounding_boxes_[i](3), depths[j] ) );

                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](2), object_bounding_boxes_[i](3), depths[j] ) );
                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](0), object_bounding_boxes_[i](3), depths[j] ) );

                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](0), object_bounding_boxes_[i](3), depths[j] ) );
                cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](0), object_bounding_boxes_[i](1), depths[j] ) );
            }
            for (int j=0; j<4; j += 2)
            {
                for (int k=1; k<4; k += 2)
                {
                    cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](j), object_bounding_boxes_[i](k), depths[0] ) );
                    cube.push_back( Eigen::Vector3d( object_bounding_boxes_[i](j), object_bounding_boxes_[i](k), depths[1] ) );
                }
            }

            for (int j=0; j<cube.size(); j++)
            {
                Eigen::Vector4d v;
                v(1) = cube[j](2);
                v(0) = (cube[j](0) - X_RES * 0.5) * v(1) * 1.1147 / X_RES;
                v(2) = (Y_RES * 0.5 - cube[j](1)) * v(1) * 0.8336 / Y_RES;
                v(3) = 1;

                v = transform * v;
                v(0) /= v(3);
                v(1) /= v(3);
                v(2) /= v(3);
                v /= 1000;

                geometry_msgs::Point point;
                point.x = v(0);
                point.y = v(1);
                point.z = v(2);
                marker.points.push_back(point);
            }

            marker_array.markers.push_back(marker);
            marker.points.clear();


            // text
            marker.id += object_bounding_boxes_.size();
            marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

            Eigen::Vector3d position = (cube[0] + cube[1] + cube[8] + cube[9]) / 4.0;
            Eigen::Vector4d v;
            v(1) = position(2);
            v(0) = (position(0) - X_RES * 0.5) * v(1) * 1.1147 / X_RES;
            v(2) = (Y_RES * 0.5 - position(1)) * v(1) * 0.8336 / Y_RES;
            v(3) = 1;

            v = transform * v;
            v(0) /= v(3);
            v(1) /= v(3);
            v(2) /= v(3);
            v /= 1000;

            marker.pose.position.x = v(0);
            marker.pose.position.y = v(1);
            marker.pose.position.z = v(2) + marker.scale.z / 2.0;
            marker.color.r = 1.;
            marker.color.g = 1.;
            marker.color.b = 1.;

            marker.text = subjects_[subject_].actions[action_].videos[video_].object_types[i];

            marker_array.markers.push_back(marker);
        }
    }

    marker_array_publisher_.publish(marker_array);
}

std::vector<int> CAD120Reader::findDepths(const Eigen::Vector4d& bounding_box, const std::vector<double>& percentages)
{
    std::vector<double> depths;
    for (int i = bounding_box(0); i < bounding_box(2); i++)
    {
        for (int j = bounding_box(1); j < bounding_box(3); j++)
        {
            if (rgbd_image_[j][i][3] != 0)
                depths.push_back( rgbd_image_[j][i][3] );  // rgbd_image_[y][x]
        }
    }

    std::vector<int> result;
    if (depths.empty())
        return result;

    std::sort(depths.begin(), depths.end());

    for (int i=0; i<percentages.size(); i++)
        result.push_back( depths[ percentages[i] * depths.size() ] );

    return result;
}

}
