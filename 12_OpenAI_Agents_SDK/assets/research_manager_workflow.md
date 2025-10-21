flowchart TD
    Start([User Query]) --> Init[Initialize ResearchManager]
    Init --> TraceID[Generate Trace ID<br/>for OpenAI Platform]
    
    TraceID --> Step1[<b>Step 1: PLAN</b><br/>_plan_searches]
    
    Step1 --> Planner[Planner Agent<br/>gpt-4.1]
    Planner --> |WebSearchPlan| PlanOutput[Generate 5-20<br/>Search Terms]
    PlanOutput --> Display1[Display: 'Will perform N searches']
    
    Display1 --> Step2[<b>Step 2: SEARCH</b><br/>_perform_searches]
    
    Step2 --> Batch{Process in batches<br/>max 5 concurrent}
    Batch --> |For each WebSearchItem| SearchLoop[Search Agent<br/>+ WebSearchTool]
    SearchLoop --> |Search Summary| Results[(Collect Results)]
    Results --> |More searches?| Batch
    
    Results --> |All done| Display2[Display: 'N/N completed']
    
    Display2 --> Step3[<b>Step 3: WRITE</b><br/>_write_report]
    
    Step3 --> Writer[Writer Agent<br/>o3-mini reasoning<br/>Runner.run_streamed]
    Writer --> StreamLoop{Stream Events Loop}
    
    StreamLoop --> |Every 5 seconds| TimeCheck{Time elapsed > 5s?}
    TimeCheck --> |Yes| UpdateMsg[Update progress message<br/>cycle through predefined messages]
    TimeCheck --> |No| StreamLoop
    UpdateMsg --> StreamLoop
    
    StreamLoop --> |Stream complete| Report[ReportData Output]
    Report --> Components{Report Components}
    
    Components --> Summary[Short Summary]
    Components --> Markdown[Markdown Report<br/>1500-2500 words]
    Components --> Questions[5 Follow-up Questions]
    
    Summary --> Display3[Display Summary]
    Markdown --> Display4[Display Full Report]
    Questions --> Display5[Display Deduplicated<br/>Follow-up Questions]
    
    Display3 --> End([Research Complete])
    Display4 --> End
    Display5 --> End
    
    style Step1 fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style Step2 fill:#F5A623,stroke:#C67D00,stroke-width:3px,color:#000
    style Step3 fill:#9B59B6,stroke:#6C3483,stroke-width:3px,color:#fff
    style Planner fill:#5DADE2,stroke:#2E5C8A,color:#fff
    style SearchLoop fill:#F8B739,stroke:#C67D00,color:#000
    style Writer fill:#BB8FCE,stroke:#6C3483,color:#fff
    style StreamLoop fill:#D7BDE2,stroke:#6C3483
    style Report fill:#52C41A,stroke:#389E0D,stroke-width:2px,color:#fff
    style End fill:#73D13D,stroke:#52C41A,stroke-width:3px,color:#fff