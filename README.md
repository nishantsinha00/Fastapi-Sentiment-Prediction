# Fastapi-Sentiment-Prediction


<!-- ABOUT THE PROJECT -->
## About The Project

There are many great README templates available on GitHub; however, I didn't find one that really suited my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![scikit-learn][scikit-learn.org]][scikit-learn-url]
* [![Tensorflow][Tensorflow.org]][Tensorflow-url]
* [![Swagger][Swagger.io]][Swagger-url]
* [![Docker][Docker.com]][Docker-url]
* [![FastAPI][fastapi.com]][fastapi-url]
* [![Python][python.org]][python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
### Prerequisites
* [![Python][python.org]][python-url]
* [![Anaconda][Anaconda.com]][Anaconda-url]

### Installation

1. Create anaconda env
    ```sh
   conda create --name FastAPIenv pyhton=3.7
   conda activate FastAPIenv
   ```
2. Clone the repo
   ```sh
   git clone https://github.com/nishantsinha00/Fastapi-Sentiment-Prediction.git
   ```
3. Install required python packages 
   ```sh
   pip install -r requirements.txt
   ```
4. Run the following command in the terminal
   ```sh
   uvicorn -app.api:app --reload
   ```
5. Go to http://localhost/8000/docs to test the API

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### or 
* Run remote docker image
    ```sh
    docker run -dp 80:80 nishantsinha00/fastapi-sentiment-prediction
   ```

<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Nishant Sinha - [@LinkedIn](https://www.linkedin.com/in/nishant-sinha-201885191/) - nishantsinha00@example.com
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Anaconda.com]: https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white
[Anaconda-url]: https://www.anaconda.com/
[scikit-learn.org]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[Tensorflow.org]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org/
[Swagger.io]: https://img.shields.io/badge/-Swagger-%23Clojure?style=for-the-badge&logo=swagger&logoColor=white
[Swagger-url]: https://swagger.io/
[Docker.com]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
[fastapi.com]: https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi
[fastapi-url]: https://fastapi.tiangolo.com/
[python.org]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
