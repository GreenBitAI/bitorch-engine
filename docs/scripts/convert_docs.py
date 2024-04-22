import argparse

from markdown_it.rules_core import replacements


class Converter:
    def __init__(self, verbose=False):
        self.active = False
        self.verbose = verbose

    def process_line(self, lines, line_num):
        self.check_start_stop(lines, line_num)
        if self.active:
            self.internal_process_line(lines, line_num)

    def check_start_stop(self, lines, line_num):
        raise NotImplementedError("Should be implemented in subclass")

    def internal_process_line(self, lines, line_num):
        raise NotImplementedError("Should be implemented in subclass")

    def replace_line(self, lines, line_num, new_line):
        if self.verbose:
            print(lines[line_num].rstrip())
            print("vvv")
            print(new_line.rstrip())
        lines[line_num] = new_line


class Replacer(Converter):
    def __init__(self, replacements, **kwargs):
        super().__init__(**kwargs)
        self.replacements = replacements
        self.keys = [pair[0] for pair in replacements]

    def check_start_stop(self, lines, line_num):
        if any(key in lines[line_num] for key in self.keys):
            self.active = True
            return
        self.active = False

    def internal_process_line(self, lines, line_num):
        for pair in self.replacements:
            new_line = lines[line_num].replace(*pair)
            self.replace_line(lines, line_num, new_line)


class SectionExtractor(Converter):
    heading_levels = ["=", "-", "~", "^"]

    def __init__(self, heading_contains, modify_by=-1, **kwargs):
        super().__init__(**kwargs)
        self.heading_contains = heading_contains
        self.modify_by = modify_by
        self.level_stop = False
        self.start_line = -1
        self.end_line = -1

    @staticmethod
    def determine_level(line):
        line = line.rstrip()
        if len(line) == 0:
            return False
        first_symbol = line[0]
        if not first_symbol in SectionExtractor.heading_levels:
            return False
        if all(c == first_symbol for c in line):
            return SectionExtractor.heading_levels.index(first_symbol)
        return False

    def check_start_stop(self, lines, line_num):
        level = self.determine_level(lines[line_num])
        if level is not False and level == self.level_stop and self.active:
            if self.verbose:
                print("Inactive:", lines[line_num])
            self.active = False
            self.end_line = line_num - 1
        if self.heading_contains in lines[line_num]:
            if self.verbose:
                print("Active:", lines[line_num])
            self.active = True
            self.start_line = line_num

    def internal_process_line(self, lines, line_num):
        level = self.determine_level(lines[line_num])
        if level is False:
            return
        if self.level_stop is False:
            self.level_stop = level
        new_line = lines[line_num].replace(
            self.heading_levels[level],
            self.heading_levels[level + self.modify_by],
        )
        self.replace_line(lines, line_num, new_line)

    def write_to_file(self, lines, output_file, mode="w"):
        with open(output_file, mode) as f:
            f.writelines(lines[self.start_line:self.end_line])


class SectionCollapseFixer(Converter):
    heading_levels = ["=", "-", "~", "^"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def check_start_stop(self, lines, line_num):
        line = lines[line_num]
        if ".. raw:: html" in line and "<details>" in lines[line_num + 2]:
            if self.verbose:
                print("Active:", line)
            self.active = True
        if "</details>" in line:
            if self.verbose:
                print("Inactive:", line)
            self.active = False
            self.replace_line(lines, line_num, "")

    def internal_process_line(self, lines, line_num):
        stripped_line = lines[line_num].strip()
        if stripped_line in ["<details>", ".. raw:: html", "</details>", "</summary>"]:
            self.replace_line(lines, line_num, "")
            return
        if stripped_line == "<summary>":
            summary_line = line_num + 1
            while lines[summary_line].strip() == "":
                summary_line += 1
            self.replace_line(lines, line_num, ".. collapse:: " + lines[summary_line])
            for i in range(line_num + 1, summary_line + 1):
                self.replace_line(lines, i, "")
            return
        self.replace_line(lines, line_num, "    " + lines[line_num])


def main(args):
    # set build_options_separate=False to integrate build options into installation part:
    build_options_separate = True
    verbose = False

    install_section = SectionExtractor("Installation", verbose=verbose)
    build_section = SectionExtractor("Build options", modify_by=-1 if build_options_separate else 0, verbose=verbose)
    section_collapse_fixer = SectionCollapseFixer(verbose=verbose)
    replacer = Replacer([
        [
            r"`docker readme <docker/README.md>`__",
            r"`docker readme <https://github.com/GreenBitAI/bitorch-engine/blob/HEAD/docker/README.md>`__"
        ],
    ], verbose=verbose)

    content = None
    with open(args.input, "r") as f:
        content = f.readlines()
    if content is None:
        return

    i = 0
    while i < len(content):
        for converter in [install_section, build_section, section_collapse_fixer, replacer]:
            converter.process_line(content, i)
        i += 1

    install_section.write_to_file(content, "docs/source/installation.rst")
    if build_options_separate:
        build_section.write_to_file(content, "docs/source/build_options.rst")
    else:
        build_section.write_to_file(content, "docs/source/installation_test.rst", mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file")
    main(parser.parse_args())
