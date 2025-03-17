# Midterm Submission

## Demo Video

[Loom Video](https://www.loom.com/share/a5230509484c48b69437ecc37ad97214)

## Background and Context

Long-form technical content creation is a time and brain intensive task. YouTube creators pour hours of effort each week into creating high-quality, engaging videos designed to teach their audience certain concepts.

One of the biggest opportunities to leverage these videos is to repurpose them for other platforms. One of the highest leverage things a creator can do is branch out to other platforms to expand their reach.

But this is time-consuming and requires a lot of context switching and a learning curve to figure out the other platform.

What if there was a tool designed specifically for YouTube creators that want to reach a professional audience and automatically converted their YouTube videos into engaging LinkedIn posts optimized for that platform?

# Task 1: Defining your Problem and Audience

As an AI Solutions Engineer, I've identified a problem: `long form content creators like YouTubers and podcasters are looking for a way to quickly and easily expand into other platforms and need a way to do this without losing their unique insights and voice`

The primary users of this application are technical educators and thought leaders who:

- Create educational video content about complex technical topics
- Need to maintain consistent presence across multiple platforms
- Want to preserve their authentic voice and teaching style
- Aim to repurpose content efficiently without sacrificing quality

There are several pain points involved in this process:

1. Manual transcription and content adaptation is time-consuming
2. Staying consistent with short form content is difficult
3. Platform-specific optimization requires expertise
4. AI platforms are hard to produce good content without a lot of tweaking and custom prompting

# Task 2: Propose a Solution

The solution is Phantra: An AI-powered content creation assistant that transforms educational video content into engaging, platform-optimized written content while preserving the creator's authentic voice and teaching style.

I will build with the following stack:

- 🤖 **LLM**: GPT-4o-mini and GPT-4o
  _Using a dual-model approach allows us to optimize for both speed and quality. GPT-4o-mini handles rapid analysis tasks while GPT-4o manages complex content generation._
- 🔢 **Embedding Models**: text-embedding-3-small (initial), sentence-transformers/**gte-Qwen2-1.5B-instruct** (fine-tuned)
  _Starting with OpenAI's embedding model for rapid prototyping, then fine-tuning an open-source model to better capture domain-specific language and teaching styles._
- 🎺 **Orchestration**: LangGraph
  _Provides the flexibility to manage complex multi-agent workflows and allows for future expansion of capabilities._
- ↗️ **Vector Store**: Qdrant
  _In-memory vector store perfect for rapid prototyping and handles our current scale well with room for growth._
- 📈 **Monitoring**: LangSmith
  _Comprehensive monitoring of our LLM application with built-in support for our LangChain-based stack._
- 📐 **Evaluation**: RAGAS
  _Industry-standard evaluation framework for assessing our RAG pipeline's performance._
- 💬 **User Interface**: Streamlit
  _Rapid development of a clean, functional interface that allows for quick iteration based on user feedback._
- 🛎️ **Serving**: Streamlit and HuggingFace
  _Provides easy deployment and hosting for both our application and fine-tuned models._

**Agentic Reasoning:** The primary agentic role in Phantra will be in the writing and editing workflow. This workflow entails the write drafting the initial LinkedIn posts and the editor refining them. The supervisor determines whether or not they then need another round of writing and revision or if they are ready for publishing.

There is a writing agent and an editing agent, the supervisor agent is responsible for passing the content back and forth between these two until it determines the content is ready to publish.

In addition to the writing/editing flow, Phantra also needs to decide which of the provided LinkedIn templates would fit best with each insight extracted from the YouTube video. There is a dedicated agent for this process, which will ingest the LinkedIn post templates, compare them to the insights passed to it from the insight extractor agent, and then assign each insight a template to pass to the writer.

# Task 3: Dealing with the Data

<aside>
📝

Task 3: Collect data for (at least) RAG and choose (at least) one external API

</aside>

I’ve identified a list of 160 high-performing LinkedIn posts we can use as example posts to pull from so we know our agent is writing high-quality content.

In addition to the templates, each YouTube transcript passed in by the user will serve as the primary data source for the LinkedIn post generation. I’m using the YouTube Transcript API to get this transcript.

The templates are stored as static data in a markdown file. There is a dedicated agent that is responsible for looking through these templates and comparing them to the insights passed to it from another agent. The agent will then assign a template from this list to each insight depending on what format fits the best and hand it off to the writer agent.

In order to get these insights, we are using LangChain's YouTube transcript loader and passing in the given URL in order to extract the transcript. The insight extractor agent will then read through the transcript and choose 7 key insights to use as the foundation for the posts.

For chunking strategy I am using a standard recursive character splitter in order to get to prototype quickly. I may experiment with semantic chunking to see if it improves metrics.

# Task 4: Building a Quick End-to-End Prototype

<aside>
📝

Task 2: Build an end-to-end RAG application using an industry-standard open-source LLM application stack and your choice of commercial off-the-shelf models

</aside>

Here’s an end to end prototype of this application built with and deployed to Streamlit

https://phantra.streamlit.app/

# Task 5: Creating a Golden Test Data Set

<aside>
📝

Task 3: Generate a synthetic test data set to baseline an initial evaluation with RAGAS

</aside>

For this evaluation, I was specifically evaluating the portion of the pipeline where we are assessing the agent’s ability to accurately extract information from the YT transcript.

| Metric                       | Value  |
| ---------------------------- | ------ |
| Context Recall               | 0.8000 |
| Context Precision            | 0.7389 |
| Faithfulness                 | 0.8755 |
| Factual Correctness          | 0.3990 |
| Response Relevancy           | 0.8747 |
| Context Entity Recall        | 0.3252 |
| Noise Sensitivity (Relevant) | 0.3848 |

We can learn several things from studying these metrics.

Our context recall is very high, this means that we are able to successfully retrieve the chunks that we need to from our transcript, we aren't leaving anything important out.

Our faithfulness is also pretty high, so our answers are factually consistent.

Factual correctness was included in order to make sure that our agent is not making up random insights that are not included in the transcript. This metric indicates we may need to adjust some things here in order to tone down the hallucinations or creativity of the LLM.

Response relevancy is another indicator that the agent is not inventing insights that were not intended by the speaker, and it performed well in this category.

Entity recall is also pretty low, but I am unsure of the level of usefulness of this metric for our use case, since it is less about purely pulling facts and more about transforming language without inventing things the speaker did not intend.

Noise sensitivity is pretty good, although could be improved. This is another indication that the LLM is likely inventing some insights on its own that the speaker did not originally say.

Overall there is a lot of room for improvement here on making sure the agent is not trying too hard to be clever or come up with its own insights and instead focuses on simply restructuring and formatting knowledge gleaned from the video. This will likely involve some extensive reworking to the prompts.

It's a bit nuanced as we are looking to reformat insights from a video, rather than directly retrieve factual information, but the metrics do indicate the LLM is taking too much liberty in crafting its own content.

# Task 6: Fine-Tuning Open-Source Embeddings

<aside>
📝

Task 6: Generate synthetic fine-tuning data and complete fine-tuning of the open-source embedding model

</aside>

Chose to fine tune **Alibaba-NLP/gte-Qwen2-1.5B-instruct** for the midterm

# Task 7: Assessing Performance

<aside>
📝

Task 7: Assess the performance of the fine-tuned agentic RAG application

</aside>

Evals were performed in this notebook: https://colab.research.google.com/drive/1TZX2XfqKFaOfNGIIgNuuU_7caZ9B-fXx#scrollTo=VD-It41p47dj

### Original Metrics

| Metric                       | Value  |
| ---------------------------- | ------ |
| Context Recall               | 0.8000 |
| Context Precision            | 0.7389 |
| Faithfulness                 | 0.8755 |
| Factual Correctness          | 0.3990 |
| Response Relevancy           | 0.8747 |
| Context Entity Recall        | 0.3252 |
| Noise Sensitivity (Relevant) | 0.3848 |

### Fine Tuned Metrics

| Metric                       | Value  |
| ---------------------------- | ------ |
| Context Recall               | 0.8014 |
| Context Precision            | 0.7494 |
| Faithfulness                 | 0.8733 |
| Factual Correctness          | 0.5900 |
| Response Relevancy           | 0.9673 |
| Context Entity Recall        | 0.3165 |
| Noise Sensitivity (Relevant) | 0.2037 |

On a second run of the evaluations, the metrics ended up being pretty similar.

In hindsight I don’t think the approach used here is necessarily the correct one. I fine tuned my embedding model on a specific transcript, but the application will be used by many users all with different transcript. Some open questions:

- Why did the fine tuned embedding model perform worse in nearly every metric?
- Would a better way be to fine tune the embedding model using a much larger data set with many different transcripts?
- Is this not the right use case for fine tuning embedding models?

Outside of this, the biggest area for improvement I see is in the prompt writing. The prompts each agent is using could be improved significantly to produce better output, and I would like to change the way I am pulling out templates as I think that strategy could be improved, which ultimately will likely come down to improving the prompts.

===

# Phantra 👻

Turn your YouTube videos into engaging LinkedIn content, automagically.

## A Note on the Actual Content

Phantra is in the very early prototype stages. The content it generates is not yet very good. I'll be working on improvements and refinements to the process but at the moment I would not recommend using this LinkedIn content as-is.

## Features

- 🎥 YouTube transcript extraction
- 🤖 Intelligent content processing using GPT-4
- 📊 Adaptive RAG for longer videos
- 💼 Professional LinkedIn post generation
- ✨ Automatic post optimization
- 📥 Easy post downloading

## Setup

1. Clone the repository
2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Launch the app
2. Enter your OpenAI API key in the sidebar
3. Paste a YouTube video URL
4. Click "Generate LinkedIn Posts"
5. Download or copy your generated posts

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages listed in `requirements.txt`

## How It Works

Phantra processes videos through several stages:

1. **Transcript Extraction**: Pulls the video transcript using YouTube's API
2. **Content Analysis**: Uses either direct processing or RAG based on transcript length
3. **Insight Generation**: Extracts key insights using GPT-4
4. **Post Creation**: Converts insights into engaging LinkedIn posts
5. **Optimization**: Refines posts for maximum engagement

## License

MIT

## Contributing

Pull requests welcome! Please ensure you test your changes before submitting.
