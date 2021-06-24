package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
	"github.com/BurntSushi/toml"
)

type header struct {
	Author   string    `toml:"author"`
	Comments bool      `toml:"comments"`
	Date     time.Time `toml:"date"`

	Title string   `toml:"title"`
	Slug  string   `toml:"slug"`
	Tags  []string `toml:"tags"`
	URL   string   `toml:"url"`

	// if true, the content will not be rendered
	Draft bool `toml:"draft"`

	// user_defined
	FeaturedImage string `toml:"featured_image"`
	Description   string `toml:"description"`
}

type extraction struct {
	imagePath string
	oldPath   string
	newPath   string
	index     int
	caption   string
}

func main() {
	if err := realMain(); err != nil {
		log.Fatalln(err)
	}
}

func realMain() error {
	var (
		flagDir  = flag.String("dir", "", "Directory to parse. By default it looks into the current dir")
		flagFile = flag.String("file", "index.md", "File to parse inside the directory.")
	)
	flag.Parse()

	if *flagDir == "" {
		return errors.New("-dir needs to be specified")
	}

	srcDir := *flagDir
	dstDir, err := os.Getwd()
	if err != nil {
		return err
	}

	files, err := ioutil.ReadDir(srcDir)
	if err != nil {
		return err
	}

	re := regexp.MustCompile(`!\[(.*?)\]\((.*?)\)`)

	// search for index file in the dir
	fileName := ""
	for _, f := range files {
		if f.Name() != *flagFile {
			continue
		}

		fileName = filepath.Join(srcDir, f.Name())
	}

	if fileName == "" {
		return fmt.Errorf("couldn't find markdown file to parse: %q", *flagFile)
	}

	out, err := ioutil.ReadFile(fileName)
	if err != nil {
		return err
	}

	content := strings.Split(string(out), "\n")

	extracted := []extraction{}

	// extract and validate data
	title := ""
	count := 0
	for i, line := range content {
		// get the tile and convert it to foo_bar to be used for image renames
		if title == "" && strings.HasPrefix(line, "# ") {
			title = strings.TrimSpace(strings.TrimPrefix(line, "#"))
		}

		if i == 0 {
			if title == "" {
				return fmt.Errorf("can't continue without a title. Have this on the first line: %q", line)
			} else {
				// remove the title as hugo creates it's own
				content[i] = ""
			}
		}

		splitted := re.FindStringSubmatch(line)
		if len(splitted) == 0 {
			continue
		}

		if len(splitted) != 3 {
			return fmt.Errorf("regex should return three matches, have: %+v", splitted)
		}

		count++

		caption := splitted[1]
		image := splitted[2]
		if image == "" {
			return fmt.Errorf("an image tag without an url detected: %q", line)
		}

		oldPath := filepath.Join(srcDir, image)
		ext := filepath.Ext(image)
		if ext == "" {
			return fmt.Errorf("found image without an extension: %q", line)
		}

		newImage := fmt.Sprintf("%s-%s%s", join(title, "-"), strconv.Itoa(count), ext)

		newPath := filepath.Join(dstDir, "static", "images", newImage)

		extracted = append(extracted, extraction{
			index:     i,
			imagePath: filepath.Join("/images", newImage),
			oldPath:   oldPath,
			newPath:   newPath,
			caption:   caption,
		})

	}

	now := time.Now().UTC()
	slug := join(title, "-")

	newName := fmt.Sprintf("%d-%d-%d-%s.markdown", now.Year(), now.Month(), now.Day(), slug)
	newFile := filepath.Join(dstDir, "content", "posts", "arslan.io", newName)

	// decode the front matter if it's exists already

	hdr := header{}

	output, err := ioutil.ReadFile(newFile)
	if err == nil {
		splitted := strings.Split(string(output), "+++\n")
		if _, err := toml.Decode(splitted[1], &hdr); err != nil {
			return err
		}
	} else {
		hdr = header{
			Author:        "Fatih Arslan",
			Comments:      true,
			Date:          now,
			Title:         title,
			Slug:          slug,
			Draft:         false,
			FeaturedImage: extracted[0].imagePath,
			URL: fmt.Sprintf("/%d/%d/%d/%s/",
				now.Year(), now.Month(), now.Day(), slug),
		}
	}

	buf := new(bytes.Buffer)
	if err := toml.NewEncoder(buf).Encode(hdr); err != nil {
		log.Fatal(err)
	}

	frontMatter := "+++\n" + buf.String() + "+++\n"
	fmt.Println("Front Matter generated in TOML:")
	fmt.Println(frontMatter)

	fmt.Printf("Found %d images to be renamed and converted to shortcode\n", len(extracted))

	for _, ex := range extracted {
		finalLine := fmt.Sprintf(`{{< figure src="%s" >}}`, ex.imagePath)
		if ex.caption != "" {
			finalLine = fmt.Sprintf(`{{< figure src="%s" caption="%s" >}}`,
				ex.imagePath, ex.caption)
		}

		if err := os.Rename(ex.oldPath, ex.newPath); err != nil {
			return err
		}

		content[ex.index] = finalLine
	}

	final := frontMatter + strings.Join(content, "\n")

	err = ioutil.WriteFile(fileName, []byte(final), 0644)
	if err != nil {
		return err
	}

	if err := os.Rename(fileName, newFile); err != nil {
		return err
	}

	fmt.Println("Done!")
	return nil
}

func join(path, sep string) string {
	splitted := strings.Split(path, " ")
	for i, s := range splitted {
		splitted[i] = strings.ToLower(s)
	}
	return strings.Join(splitted, sep)
}
