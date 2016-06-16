#include <pcml/data/skeleton_stream.h>

#include <visualization_msgs/MarkerArray.h>

#include <stdio.h>


namespace pcml
{

std::vector<std::string> SkeletonStream::jointNamesWholeBody()
{
    static const std::vector<std::string> whole_body =
    {
        "head",
        "neck",
        "torso",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_shoulder",
        "right_elbow",
        "right_hand",
        "left_hip",
        "left_knee",
        "left_foot",
        "right_hip",
        "right_knee",
        "right_foot",
    };

    return whole_body;
}

std::vector<std::string> SkeletonStream::jointNamesUpperBody()
{
    static const std::vector<std::string> upper_body =
    {
        "head",
        "neck",
        "torso",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_shoulder",
        "right_elbow",
        "right_hand",
    };

    return upper_body;
}


// SkeletonStream
SkeletonStream::SkeletonStream()
    : SkeletonStream(jointNamesWholeBody())
{
}

SkeletonStream::SkeletonStream(const std::vector<std::string>& joint_names)
    : joint_names_(joint_names)
    , finished_(true)
{
}

bool SkeletonStream::getSkeleton(Eigen::Matrix3Xd& skeleton)
{
    return false;
}

void SkeletonStream::setVisualizationTopic(ros::NodeHandle nh, const std::string &topic)
{
    marker_array_publisher_ = nh.advertise<visualization_msgs::MarkerArray>(topic, 1);
}

void SkeletonStream::renderSkeleton()
{
    Eigen::Matrix3Xd skeleton;
    if (!getSkeleton(skeleton))
        return;

    renderSkeleton(skeleton);
}

void SkeletonStream::renderSkeleton(const Eigen::Matrix3Xd& skeleton)
{
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker;

    marker.header.frame_id = getFrameId();
    marker.header.stamp = ros::Time::now();

    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.id = 0;

    // spheres
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.scale.x = 0.1; // 10cm
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.r = 0.5;
    marker.color.g = 0.5;
    marker.color.b = 0.5;
    marker.color.a = 1.0;
    marker.pose.position.x = 0;
    marker.pose.position.y = 0;
    marker.pose.position.z = 0;
    for (int i=0; i<joint_names_.size(); i++)
    {
        geometry_msgs::Point point;

        marker.ns = joint_names_[i];

        point.x = skeleton(0, i);
        point.y = skeleton(1, i);
        point.z = skeleton(2, i);
        marker.points.push_back(point);
    }
    marker_array.markers.push_back(marker);

    // texts
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.id = 1;
    marker.scale.x = 0.1; // 10cm
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    for (int i=0; i<joint_names_.size(); i++)
    {
        marker.ns = joint_names_[i];

        marker.text = joint_names_[i];
        marker.pose.position.x = skeleton(0, i);
        marker.pose.position.y = skeleton(1, i);
        marker.pose.position.z = skeleton(2, i);

        marker_array.markers.push_back(marker);
    }

    marker_array_publisher_.publish(marker_array);
}


// SkeletonFileStreamAbstract
SkeletonFileStreamAbstract::SkeletonFileStreamAbstract()
    : SkeletonFileStreamAbstract(jointNamesWholeBody())
{
}

SkeletonFileStreamAbstract::SkeletonFileStreamAbstract(const std::vector<std::string> &joint_names)
    : SkeletonStream(joint_names)
    , is_realtime_(true)
{
}

bool SkeletonFileStreamAbstract::getSkeleton(Eigen::Matrix3Xd& skeleton)
{
    return false;
}


// SkeletonFileStream
SkeletonFileStream::SkeletonFileStream()
    : SkeletonFileStream(jointNamesWholeBody())
{
}

SkeletonFileStream::SkeletonFileStream(const std::vector<std::string> &joint_names)
    : SkeletonFileStreamAbstract(joint_names)
{
    fp_ = NULL;
}

SkeletonFileStream::~SkeletonFileStream()
{
    if (fp_ != NULL)
        fclose(fp_);
}

void SkeletonFileStream::startReadFrames()
{
    fp_ = fopen(filename_.c_str(), "r");
    if (fp_ == NULL)
        return;

    fscanf(fp_, "%d", &file_num_joints_);
    file_joint_index_map_.resize(file_num_joints_);

    std::vector<char> requested_joint_name_exist(joint_names_.size(), false);

    for (int i=0; i<file_num_joints_; i++)
    {
        static char joint_name[128];
        fscanf(fp_, "%lf", joint_name);

        file_joint_names_.push_back(joint_name);

        const int idx = std::find(joint_names_.begin(), joint_names_.end(), joint_name) - joint_names_.begin();
        if (idx != joint_names_.size())
        {
            // requested joint is found in the saved file
            file_joint_index_map_[i] = idx;
            requested_joint_name_exist[idx] = true;
        }
        else
            file_joint_index_map_[i] = -1;
    }

    // all requested joint names should exist in the file
    for (int i=0; i<joint_names_.size(); i++)
    {
        if (!requested_joint_name_exist[i])
        {
            fclose(fp_);
            fp_ = NULL;
            return;
        }
    }
}

bool SkeletonFileStream::getSkeleton(Eigen::Matrix3Xd& skeleton)
{
    if (fp_ == NULL)
        return false;

    skeleton.resize(Eigen::NoChange, joint_names_.size());

    for (int i=0; i<file_num_joints_; i++)
    {
        Eigen::Vector3d v;
        if (fscanf(fp_, "%lf%lf%lf", &v(0), &v(1), &v(2)) != 3)
        {
            fclose(fp_);
            fp_ = NULL;
            return false;
        }

        if (file_joint_index_map_[i] != -1)
            skeleton.col( file_joint_index_map_[i] ) = v;
    }

    return true;
}


// SkeletonCAD120Stream
const std::map<std::string, std::string> SkeletonCAD120Stream::joint_name_conversion_map_ =
{
    {"head", "HEAD"},
    {"neck", "NECK"},
    {"torso", "TORSO"},
    {"left_shoulder", "LEFT_SHOULDER"},
    {"left_elbow", "LEFT_ELBOW"},
    {"right_shoulder", "RIGHT_SHOULDER"},
    {"right_elbow", "RIGHT_ELBOW"},
    {"left_hip", "LEFT_HIP"},
    {"left_knee", "LEFT_KNEE"},
    {"right_hip", "RIGHT_HIP"},
    {"right_knee", "RIGHT_KNEE"},
    {"left_hand", "LEFT_HAND"},
    {"right_hand", "RIGHT_HAND"},
    {"left_foot", "LEFT_FOOT"},
    {"right_foot", "RIGHT_FOOT"},
};

SkeletonCAD120Stream::SkeletonCAD120Stream(const std::string& directory)
    : SkeletonCAD120Stream(directory, jointNamesWholeBody())
{
}

SkeletonCAD120Stream::SkeletonCAD120Stream(const std::string &directory, const std::vector<std::string> &joint_names)
    : SkeletonFileStreamAbstract(joint_names)
    , reader_(directory)
{
}

bool SkeletonCAD120Stream::getSkeleton(Eigen::Matrix3Xd& skeleton)
{
    if (!reader_.readNextFrame())
    {
        finished_ = true;
        return false;
    }

    skeleton.resize(Eigen::NoChange, joint_names_.size());
    for (int i=0; i<joint_names_.size(); i++)
    {
        std::map<std::string, std::string>::const_iterator it = joint_name_conversion_map_.find(joint_names_[i]);
        if (it == joint_name_conversion_map_.end())
        {
            // joint name could not be found in conversion map
            fprintf(stderr, "ERROR: %s is not found in CAD120 joint name conversion map\n", joint_names_[i].c_str());
            return false;
        }

        const std::string cad120_joint_name = it->second;
        Eigen::Vector3d position;

        if (!reader_.getJointPosition(cad120_joint_name, position))
        {
            // joint name is not defined in CAD120
            fprintf(stderr, "ERROR: %s is not defined in CAD120\n", cad120_joint_name.c_str());
            return false;
        }

        skeleton.col(i) = Eigen::Vector3d(position(0), position(2), position(1)) / 1000.;
    }

    return true;
}

int SkeletonCAD120Stream::numSubjects()
{
    return reader_.numSubjects();
}

int SkeletonCAD120Stream::numActions(int subject)
{
    return reader_.numActions(subject);
}

int SkeletonCAD120Stream::numVideos(int subject, int action)
{
    return reader_.numVideos(subject, action);
}

void SkeletonCAD120Stream::startReadFrames(int subject, int action, int video)
{
    finished_ = false;
    reader_.startReadFrames(subject, action, video);
}


// SkeletonRealtimeStream
SkeletonRealtimeStream::SkeletonRealtimeStream(const ros::NodeHandle& nh)
    : SkeletonRealtimeStream(nh, jointNamesWholeBody())
{
}

SkeletonRealtimeStream::SkeletonRealtimeStream(const ros::NodeHandle& nh, const std::vector<std::string>& joint_names)
    : SkeletonStream(joint_names)
    , nh_(nh)
{
    finished_ = false;
}

bool SkeletonRealtimeStream::getSkeleton(Eigen::Matrix3Xd& skeleton)
{
    const int id = getUserId();
    if (id == -1)
        return false;

    const std::string id_string = std::to_string(id);

    skeleton.resize(Eigen::NoChange, joint_names_.size());
    for (int i=0; i<joint_names_.size(); i++)
    {
        tf::StampedTransform transform;
        try
        {
            listener_.lookupTransform("camera_depth_frame", joint_names_[i] + "_" + id_string, ros::Time(0), transform);
            tf::Vector3 p = transform.getOrigin();
            skeleton.col(i) = Eigen::Vector3d(p.x(), p.y(), p.z());
        }
        catch (tf::TransformException ex)
        {
            return false;
        }
    }

    return true;
}

int SkeletonRealtimeStream::getUserId()
{
    tf::StampedTransform transform;

    for (int id=1; id<=MAX_USER_ID; id++)
    {
        const std::string id_string = std::to_string(id);

        try
        {
            listener_.lookupTransform("camera_depth_frame", joint_names_[0] + "_" + id_string,
                                   ros::Time(0), transform);
            return id;
        }
        catch (tf::TransformException ex)
        {
        }
    }

    return -1;
}

}
