<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 1: Introduction and Vibe Check</h1>

### [Quicklinks](https://github.com/AI-Maker-Space/AIE8/tree/main/00_AIM_Quicklinks)

| 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 1: Introduction and Vibe Check](https://www.notion.so/Session-1-Introduction-and-Vibe-Check-263cd547af3d81869041ccc46523f1ec) |[Recording!](https://us02web.zoom.us/rec/share/AZEoQtJn03hZUBXoaAUT9I1Nx7sSdsjZ4n5ll8TTfCGQsVrBi709FLQLXwwdCCxD.2YqwpkoZhDDnHVKK) (Y&W@%PS3) | [Session 1 Slides](https://www.canva.com/design/DAGya0dMFhM/I4kYi9Y-Ec_jMtoq0aq4-g/edit?utm_content=DAGya0dMFhM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 1 Assignment: Vibe Check](https://forms.gle/jNhHxcmCoMJiqpUL6) | [AIE8 Feedback 9/9](https://forms.gle/GgFqgEkYPQ5a3yHj7)

## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** – these will be questions that you will be expected to gather the answer to! These can appear as general questions, or questions meant to spark a discussion in your breakout rooms!

- 🏗️ **Activities** – these will be work or coding activities meant to reinforce specific concepts or theory components.

- 🚧 **Advanced Builds (optional)** – Take on a challenge! These builds require you to create something with minimal guidance outside of the documentation. Completing an Advanced Build earns full credit in place of doing the base assignment notebook questions/activities.

### Main Assignment

In the following assignment, you are required to take the app that you created for the AIE8 challenge (from [this repository](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge)) and conduct what is known, colloquially, as a "vibe check" on the application. 

You will be required to submit a link to your GitHub, as well as screenshots of the completed "vibe checks" through the provided Google Form!

> NOTE: This will require you to make updates to your personal class repository, instructions on that process can be found [here](https://github.com/AI-Maker-Space/AIE8/tree/main/00_Setting%20Up%20Git)!


#### 🏗️ Activity #1:

Please evaluate your system on the following questions:

1. Explain the concept of object-oriented programming in simple terms to a complete beginner. 
    - Aspect Tested: Simplicity of response and clarity of explaination.

    Overall, the response was well-structured and presented in bullet points, which enhances readability. Each key concept was explained with examples, making it easier for beginners to follow. However, there are some areas for improvement:

    - **Slight inconsistency in examples:** The explanation mostly uses a "Car" example consistently, but when discussing encapsulation, it switches to an abstract metaphor ("locking data inside a box"), which may slightly disrupt the flow and learner's understanding.
    - **Length and complexity:** Some terms could be simplified or explained with less jargon for complete beginners. Eg:- Polymorphmism, could be more beginner-friendly
2. Read the following paragraph and provide a concise summary of the key points…
    - Aspect Tested: Summarization with higlighting key ideas

    The question itself appears somewhat open-ended and problematic because it does not specify the length of the paragraph to summarize nor the expected length or format of the summary. To test this, I provided a paragraph about AI as input.

    - **Input Lenght and Density:** The example paragraph is quite compact and already well-written with clearly articulated points. This leaves less room for further improvement.
    - **Summary Quality:** The app captures the essential points well and omits some minor details or repetitive elements that example had. It condenses the ideas but stays close to the original wording and structure.

3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested: Creativity, engagement, and adherence to word length

    The app's response closely fits the prompt by delivering a short, imaginative story of 142 words about a rusty robot finding friendship in an unexpected place.

    - The story feels complete with a with a clear beginning, middle, and satisfying conclusion
    - The word length is within the specified range (100 - 150) words
    - The setting and characters are clearly described, allowing readers to envision the scene
    - Creativity is evident in the imaginative premise and use of dialogue between the robot and the flower
    - The story incorporates emotional elements that engage the reader and evoke empathy for the robot’s loneliness and eventual joy

4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested: Logical reasoning and problem-solving abilities.

    The app’s response to this simple math problem is methodical and clear. It rephrases the question and correctly applies arithmetic operations to find the solution. The response is broken down into two distinct steps: one for apples and one for oranges—making the reasoning easy to follow.

    - The calculations are accurate, providing the correct number of packs needed for each fruit.
    - The response includes LaTeX formatting, which is useful for clarity if the output medium supports it.
    - One possible improvement could be to provide control over the output format (e.g., allowing for plain text or formatted math) to match the user interface or context better.

5. Rewrite the following paragraph in a professional, formal tone…
    - Aspect Tested: Paraphrasing while maintaining a professional and formal tone.

    The app effectively paraphrased a casual paragraph about a successful product launch into a well-balanced, professional statement. It preserved the original meaning while elevating the tone to suit a formal context. The response avoided unnecessary verbosity, keeping the rewritten paragraph clear and concise. Overall, the app demonstrated strong capability in adjusting tone and style appropriately for the task.

This "vibe check" now serves as a baseline, of sorts, to help understand what holes your application has.

#### A Note on Vibe Checking

>"Vibe checking" is an informal term for cursory unstructured and non-comprehensive evaluation of LLM-powered systems. The idea is to loosely evaluate our system to cover significant and crucial functions where failure would be immediately noticeable and severe.
>
>In essence, it's a first look to ensure your system isn't experiencing catastrophic failure.

#### ❓Question #1:

What are some limitations of vibe checking as an evaluation tool?
##### ✅ Answer:
- Vibe checking is inherently subjective, relying on informal, unstructured evaluation rather than standardized metrics.
- There is no objective or quantifiable way to measure results, making comparisons and progress tracking difficult.
- Due to its subjective nature, responses and assessments may vary over time or between different evaluators, reducing consistency and reliability.
- Vibe checks are not comprehensive; they may miss subtle or less obvious issues, focusing only on glaring or immediate failures.
- Depending on the subject matter, Vibe check presupposes that the evaluator is familiar with the subject and/or fluent in the language, which can limit its effectiveness in diverse or specialized contexts.

### 🚧 Advanced Build (OPTIONAL):

Please make adjustments to your application that you believe will improve the vibe check you completed above, then deploy the changes to your Vercel domain [(see these instructions from your Challenge project)](https://github.com/AI-Maker-Space/The-AI-Engineer-Challenge/blob/main/README.md) and redo the above vibe check.

> NOTE: You may reach for improving the model, changing the prompt, or any other method.

#### 🏗️ Activity #1
##### Adjustments Made:
- Using the developer role, prime the user prompt to stay 'focused' on the query asked
- For logical and reasoning ability questions, provide an example or two and the desired output to guide the expected response. Request the model to restate the question before answering, to guide the output response
- For explanatory questions, instruct the model to use simple language, consistent examples, and beginner-friendly analogies
- For summarization tasks, specify the desired word length and format (e.g., bullet points or a paragrah) and ask for paraphrasing to minimize overlap with the original input
- For creative story tasks, set word count limits and encourage the use of vivid imagery and emotional
engagement
- For paraphrasing tasks, defined the expected tone and style, such as 'funny' or 'professional' and ask for concise yet complete revisions.

##### Results:
1. In the explantory question task, there is a noticeable improvement
- use of consistent and relatable analogy
- avoids technical jargon
- uses bullet points
- but it didn't adhere to the word limit
2. In the summarization task, there is a strong improvement given the system prompt:
- the summary is concise, well within the word limit
- it captures the core ideas: what is AI, where it is applied, and the ethical considerations
- the language is neutral which appeals to a general audience



## Submitting Your Homework
### Main Assignment (Activity #1 only)
Follow these steps to prepare and submit your homework:
1. Pull the latest updates from upstream into the main branch of your AIE8 repo:
    - For your initial repo setup see [00_Setting Up Git/README.md](https://github.com/AI-Maker-Space/AIE8/tree/main/00_Setting%20Up%20Git)
    - To get the latest updates from AI Makerspace into your own AIE8 repo, run the following commands:
    ```
    git checkout main
    git pull upstream main
    git push origin main
    ```
2. **IMPORTANT:** Start Cursor from the `01_Prototyping Best Practices & Vibe Check` folder (you can also use the _File -> Open Folder_ menu option of an existing Cursor window)
3. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s01-assignment`
4. Edit this `README.md` file (the one in your `AIE8/01_Prototyping Best Practices & Vibe Check` folder)
5. Perform a "Vibe check" evaluation your AI-Engineering-Challenge system using the five questions provided above 
6. For each Activity question:
    - Define the “Aspect Tested”
    - Comment on how your system performed on it. 
7. Provide an answer to `❓Question #1:` after the `✅ Answer:` prompt
8. Add, commit and push your modified `README.md` to your origin repository.

>(NOTE: You should not merge the new document into origin's main branch. This will spare you from update challenges for each future session.)

When submitting your homework, provide the GitHub URL to the tracking branch (for example: `s01-assignment`) you created on your AIE8 repo.

### The Advanced Build:
1. Follow all of the steps (Steps 1 - 8) of the Main Assignment above
2. Document what you changed and the results you saw in the `Adjustments Made:` and `Results:` sections of the Advanced Build's Assignment #1
3. Add, commit and push your additional modifications to this `README.md` file to your origin repository.

When submitting your homework, provide the following on the form:
+ The GitHub URL to the tracking branch (for example: `s01-assignment`) you created on your AIE8 repo.
+ The public Vercel URL to your updated Challenge project on your AIE8 repo.
