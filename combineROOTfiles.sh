while getopts i:s:t:o:m: flag
do
    case "${flag}" in
        i) dir_in=${OPTARG};;
        s) station=${OPTARG};;
        t) type=${OPTARG};;
        o) dir_out=${OPTARG};;
        m) dir_sim=${OPTARG};;
    esac
done

if [[ "$dir_in" != */ ]]
then
    dir_in+="/"
fi

if [[ "$dir_out" != */ ]]
then
    dir_out+="/"
fi

file_in=${dir_in}${type}"_s"${station}"_"*".root"

if [[ "$file_in" == _train.root ]]
then
    file_out=${dir_out}${type}"_s"${station}"_train.root"
    if [[ "$dir_sim" != "" ]]
    then
        sim_in=${dir_sim}${type}"_sim_s"${station}"_"*"_train.root"
    fi
elif [[ "$file_in" == _test.root ]]
then
    file_out=${dir_out}${type}"_s"${station}"_test.root"
    if [[ "$dir_sim" != "" ]]
    then
        sim_in=${dir_sim}${type}"_sim_s"${station}"_"*"_test.root"
    fi
else
    file_out=${dir_out}${type}"_s"${station}".root"
    if [[ "$dir_sim" != "" ]]
    then
        sim_in=${dir_sim}${type}"_sim_s"${station}"_"*".root"
    fi
fi

# Must have ROOT to use the "hadd" command to merge .root files
hadd ${file_out} ${file_in} ${sim_in}
