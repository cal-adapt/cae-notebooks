## Summary of changes
[What's changed in this PR?]

## Link to corresponding Jira ticket(s)
[What Jira ticket(s) describe further context/detail for this PR?]

## Naming & Organization
- [ ] Notebook name follows conventions:
  - [ ] Avoids acronyms and jargon where possible
  - [ ] References primary use of the notebook
  - [ ] Uses Action + Objective format (e.g., "Calculate_Heat_Index", "Calculate_Annual_Trends")
  - [ ] Prioritizes clarity over brevity (within reason)
- [ ] Notebook placed in appropriate directory for maximum usability
- [ ] All referenced guides/documentation updated:
  - [ ] README updated
  - [ ] Navigation Guide updated
  - [ ] Other relevant documentation updated

## Content & Documentation
- [ ] Introduction clearly explains the purpose of the analysis
- [ ] Expected outcome/use of the notebook included
  - [ ] User-focused question/example provided showing how to use the notebook
- [ ] Key takeaway messages clearly stated (what users will learn)
- [ ] Total runtime of the notebook listed
- [ ] Incorporates references to appropriate Guidance materials
- [ ] Error messages included for areas with common user errors

## Markdown Quality
- [ ] Markdown reviewed and vetted by another person
- [ ] No typos in markdown cells
- [ ] Markdown formatting correct and readable

## Runtime Information
- [ ] Overall runtime on AE JupyterHub documented
- [ ] Overall runtime on pcluster documented (if applicable)

## Output Management
- [ ] Output from cells removed before committing. If outputs are kept:
  - [ ] Exceptions documented (e.g., perhaps long-running functions that should show intended outcome for user)
  - [ ] Security and other impacts of keeping outputs considered
  - [ ] Alternative documentation methods explored (e.g., markdown descriptions)

## Code Quality
- [ ] Notebook linted with:
  - [ ] black
  - [ ] isort
  - [ ] ruff
