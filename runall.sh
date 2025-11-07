command="./instance.py"

for file in data/testing/*
do
    echo $file
    $command "$file"
done

for file in data/*
do
    echo $file
    $command "$file"
done
