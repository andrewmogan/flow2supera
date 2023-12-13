# Contribution Guidelines

The basic steps to contributing to a Github project apply here:
1. Create a [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of this repository.
2. Clone your forked repository to your local machine.
3. Make your changes on a new branch, including documentation (see below).
4. Run the tests to make sure everything works as expected.
5. Push your changes to your forked repository.
6. Submit a pull request (PR) to the main repository.

# Opening Issues
If you see that something isn't working as intended, or would like to request a new feature or improvement, open up a [Github Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues). Please include a detailed description of the problem and include appropriate tags.

# Code Contributions

## Readability
Code readability is criminally undervalued in the physics world. Writing clean, readable code can help both you and those reading your code to understand, use, and modify it. Python's famous [PEP 8](https://peps.python.org/pep-0008/#pet-peeves) covers this extensively, but here are a few guidelines to keep in mind when contributing to this repository:
- **Use descriptive names:** Names for variables, functions, and classes should describe their purpose. Using descriptive variable names can go a long way toward helping others understand what you're doing by eliminating guesswork ("hang on, what exactly is `x` here?"). To that end, single-letter variable names are **strongly** discouraged. Even iterators can often benefit from an extra letter or two.  
- **Use comments _sparingly_:** Counterintiutively, comments can sometimes reduce readability if abused. Comments can easily become deprecated or wrong when the code around them changes without also updating the comment. Descriptive naming (see previous point) can often eliminate the need for comments.
- **Avoid hard-coding:** Instead, use a constant value with a descriptive name and assign variables to that value where appropriate. 
- **Keep lines short:** Try to keep each line of code to roughly 80 characters or fewer. Break code up over multiple lines if necessary. 
- **Keep nesting to a minimum:** If you find yourself more than 3 indentations deep in your code (and even 3 is pushing it), consider "denesting," as in this example [here](https://testing.googleblog.com/2017/06/code-health-reduce-nesting-reduce.html).
- **Use consistent casing:** In flow2supera, classes and class methods are named using `PascalCase`, as in `TrajectoryToParticle`, while module names use `snake_case`.

The table below provides some examples illustrating the above points.

| Less readable | More readable |
|----------|----------|
| x | x_position |
| for t in trks  | for track in tracks  |
| # Define a function to add two numbers <br> def func(x, y) | def add_two(track, crthit) |
| # Set mass <br> m = 135 | PION_MASS = 135 <br> particle_mass = PION_MASS | 

## Tests

As of this writing, `flow2supera` does not have a unit test framework, probably because its lazy author lacks formal education and experience in such matters. Until such a framework exists, users are expected to test their own code before submitting a PR. Here are some simple checks to keep in mind:
- The code should build successfully using the _exact_ same command listed in the README (up to a `--user` flag if applicable). 
- The executable `bin/run_flow2supera.py` should run without producing errors when given a proper input file. 

## Documentation
Like readability, documentation is often considered an afterthought, an inconvenience that makes us think "maybe I'll get around to that if I feel like it." But without documentation, future users (and our future selves) can often struggle to understand what the code is doing and how to use it. Any function you write should have a docstring, no matter how "obvious" you think it is. Use the template below to stay consistent with the rest of this repository.

### Docstring Template

TODO Create a template a put it here once a format is settled. 
