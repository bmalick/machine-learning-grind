if [ -f diabetes.cvs ]; then
    echo "File already exists"
else
    wget https://storage.googleapis.com/kagglesdsdata/datasets/228/482/diabetes.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250126%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250126T165759Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=72f0403e43bb650642d737e19624303bc4e981a6068d7cd7afb1cb7ea0df1913141327db508c29845bc16f0c4d9007bd20efc0b06b8c61faf323df5d955fe39b840bd05896fa9ad4a6fd5bfa5e714fdb4785dbdb44e6942ceeda890b5de1e63a99831d3c1d460a9a8b3c1ea8a5d58dda6a6f26c6a2070d743c3e9852f22dde50b57fc3e8b07e49f803dc00978c9b8b6bb736b9c7e74572300de39fb9e033cb3432eb40db8d916c2b624f8b8478c680d18bbce8f55fba050a1a849cc6d90aaa08cce9933135b5a829ddc48ca74cd29ecd58decfe97f117c8f63a1c7a4711a3d92e3e393cf3fca35d783c74e43701bdee4bd60227caa74271bdf045f5153b2281a

fi