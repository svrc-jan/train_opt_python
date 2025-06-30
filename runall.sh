command="./instance.py"

for file in data/testing/*
do
    echo $file
    $command "$file"
done

for file in data/phase1/*
do
    echo $file
    $command "$file"
done

for file in data/phase2/*
do
    echo $file
    $command "$file"
done
