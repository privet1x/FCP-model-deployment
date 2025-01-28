#Link for Fake Currency Detection analysis article - https://thecleverprogrammer.com/2020/09/29/fake-currency-detection-with-machine-learning/

Docker file for app we run command from root directory: docker build -t asi-app -f Dockerfile .

after dockerfile has been builded we can run container using: docker run -p 8000:8000 asi-app

for ansible we run docker-compose up --build -d

then we do: docker exec -it fcp-model-deployment-target_host-1 bash

and write there 2 commands one by one

dockerd &

usermod -aG docker ansible

after that we exit container using command: exit

after that we access another container

docker exec -it fcp-model-deployment-target_host-1 bash

and run

ansible-playbook -i inventory.yml playbook.yml

after that your application will be working fine if you DO NOT HAVE problems with dependencies