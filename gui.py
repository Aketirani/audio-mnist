import datetime
import os
import subprocess
import sys
import tkinter as tk
from tkinter import Label, Menu, Toplevel, filedialog, messagebox

from PIL import Image, ImageTk

from src.setup import Setup


class GUI:
    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI application

        :param root: tk.Tk, root Tkinter window
        """
        self.colors = {
            "gui": "light grey",
            "title": "navy blue",
            "run": "green",
            "exit": "red",
            "fg": "white",
            "disp": "black",
        }

        self.fonts = {
            "title": ("Helvetica", 12, "bold"),
            "checkbox": ("Helvetica", 10, "bold", "underline"),
            "button": ("Helvetica", 10, "bold"),
        }

        self.title = "UTILIZING XGBOOST IN VOICE GENDER CLASSIFICATION"
        self.initial_text = "PLEASE SELECT THE BOXES, AND CLICK RUN... ✔️"
        self.nothing_text = "NO BOXES WERE SELECTED, PLEASE TRY AGAIN... ⚠️"
        self.wait_display = "PROCESSING, PLEASE WAIT... ⏳"

        self.checkbox_labels = [
            "DATA PREPARATION",
            "FEATURE ENGINEERING",
            "DATA SPLITTING",
            "MODEL TUNING",
            "MODEL TRAINING",
            "MODEL PREDICTION",
            "DATA TO POSTGRESQL",
        ]

        self.tooltip_texts = [
            "EXTRACT, TRANSFORM AND LOAD AUDIO DATASET",
            "PERFORM COMPREHENSIVE FEATURE ENGINEERING",
            "SPLIT DATASET INTO TRAINING, VALIDATION, AND TEST SETS",
            "TUNE MODEL HYPERPARAMETERS FOR PERFORMANCE OPTIMIZATION",
            "TRAIN XGBOOST MODEL USING TRAINING SET",
            "EVALUATE MODEL PERFORMANCE USING TEST SET",
            "WRITE CSV DATASETS TO TABLES IN POSTGRESQL",
        ]

        self.SU = Setup("config.yaml")
        self.config_file = self.SU.read_config()
        self.root = root
        root.title("AudioMNIST")
        root.geometry(f"500x750+10+10")
        root.configure(bg=self.colors["gui"])
        self.create_menu_bar()
        self.create_widgets()
        self.redirect_output()
        self.history_log = []

    def create_menu_bar(self) -> None:
        """
        Create a menu bar
        """
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save", command=self.save_output)
        file_menu.add_command(label="History", command=self.show_history)

        data_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Data", menu=data_menu)
        data_menu.add_command(
            label="Columns",
            command=lambda: self.show_help_boxes(
                self.config_file["text"]["cols"], "Columns"
            ),
        )

        visuals_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visuals", menu=visuals_menu)
        visuals_menu.add_command(label="Plots", command=self.show_plots)

        help_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(
            label="About",
            command=lambda: self.show_help_boxes(
                self.config_file["text"]["about"], "About"
            ),
        )

    def create_widgets(self) -> None:
        """
        Create various widgets for the GUI
        """
        self.create_label("", pady=0)
        self.create_label(self.title, fg=self.colors["title"], font=self.fonts["title"])
        self.create_label("", pady=0)
        self.create_image_label()
        self.create_label("", pady=0)
        self.create_checkboxes()
        self.create_label("", pady=0)
        self.output_text = self.create_output_text()
        self.create_label("", pady=0)
        self.create_button(
            "RUN",
            self.run_pipeline,
            self.colors["run"],
            self.SU.set_img_path(),
            self.config_file["image"]["run"],
        )
        self.create_label("", pady=0)
        self.create_button(
            "EXIT",
            self.exit_app,
            self.colors["exit"],
            self.SU.set_img_path(),
            self.config_file["image"]["exit"],
        )

    def show_tooltip(self, event, text) -> None:
        """
        Show a tooltip with the provided text

        :param event: The click event that triggered the tooltip
        :param text: The text to display in the tooltip
        """
        tooltip = Toplevel(self.root)
        tooltip.wm_overrideredirect(True)

        x, y, _, _ = event.widget.bbox("insert")
        x += event.widget.winfo_rootx() + 25
        y += event.widget.winfo_rooty()
        tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tooltip, text=text, background="yellow", relief="solid", borderwidth=1
        )
        label.pack()

        self.root.bind("<Button-1>", lambda e, tooltip=tooltip: tooltip.destroy())

    def show_help_boxes(self, filename: str, title: str) -> None:
        """
        Display a help box in a separate window based on the provided filename and title

        :param filename: str, filename of the text file
        :param title: str, title for the window
        """
        help_window = Toplevel(self.root)
        help_window.title(title)
        help_window.geometry("500x500")

        frame = tk.Frame(help_window)
        frame.pack(fill=tk.BOTH, expand=True)

        help_text = tk.Text(frame, wrap=tk.WORD, width=40, height=20)
        help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame, command=help_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        help_text.config(yscrollcommand=scrollbar.set)
        help_path = os.path.join(self.SU.set_txt_path(), filename)
        with open(help_path, "r", encoding="utf-8") as help_file:
            help_contents = help_file.read()

        lines = help_contents.splitlines()
        section_number = None

        for line in lines:
            line = line.strip()
            if line.startswith("**") and line.endswith("**"):
                help_text.insert(tk.END, line.strip("**") + "\n", "bold")
            elif line.startswith("**"):
                section_number = line.strip("**")
                help_text.insert(tk.END, section_number + " ", "section_number")
                help_text.insert(
                    tk.END, line.strip("**")[len(section_number) :] + "\n", "bold"
                )
            else:
                if section_number:
                    help_text.insert(tk.END, line + "\n", "section_content")
                else:
                    help_text.insert(tk.END, line + "\n")

        help_text.tag_configure("bold", font=("Helvetica", 12, "bold"))
        help_text.tag_configure("section_number", font=("Helvetica", 12, "bold"))
        help_text.tag_configure("section_content", font=("Helvetica", 12))

    def create_label(self, text: str, **kwargs: any) -> None:
        """
        Create a label widget

        :param text: str, the label text
        :param kwargs: additional keyword arguments for label creation
        """
        kwargs["bg"] = self.colors["gui"]
        label = Label(self.root, text=text, **kwargs)
        label.pack()

    def create_image_label(self) -> None:
        """
        Create a label widget to display a resized image
        """
        image_path = os.path.join(
            self.SU.set_img_path(), self.config_file["image"]["logo"]
        )
        image_orginal = Image.open(image_path)
        image_resized = image_orginal.resize((250, 150), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image_resized)
        image_label = Label(self.root, image=photo)
        image_label.image = photo
        image_label.pack()

    def create_checkboxes(self) -> None:
        """
        Create checkbox widgets for various pipeline steps with corresponding tooltips
        """
        checkboxes_frame = tk.Frame(self.root, bg=self.colors["gui"])
        checkboxes_frame.pack()

        self.checkbox_vars = []
        for label, tooltip_text in zip(self.checkbox_labels, self.tooltip_texts):
            checkbox_frame = tk.Frame(checkboxes_frame, bg=self.colors["gui"])
            checkbox_frame.grid(sticky="w", padx=20)

            var = tk.BooleanVar()
            self.checkbox_vars.append(var)
            checkbox = tk.Checkbutton(
                checkbox_frame,
                text=label,
                variable=var,
                bg=self.colors["gui"],
                fg=self.colors["disp"],
                font=self.fonts["checkbox"],
            )
            checkbox.grid(row=0, column=0, sticky="w")

            info_path = os.path.join(
                self.SU.set_img_path(), self.config_file["image"]["info"]
            )
            info_original = Image.open(info_path)
            info_resized = info_original.resize((15, 15), Image.LANCZOS)
            info_image = ImageTk.PhotoImage(info_resized)
            info_label = Label(checkbox_frame, image=info_image, bg=self.colors["gui"])
            info_label.image = info_image
            info_label.grid(row=0, column=1)
            info_label.bind(
                "<Button-1>",
                lambda event, text=tooltip_text: self.show_tooltip(event, text),
            )

    def create_output_text(self, **kwargs: any) -> tk.Text:
        """
        Create a text widget for displaying output

        :param width: int, the width of the text widget
        :param height: int, the height of the text widget
        :param kwargs: additional keyword arguments for text widget creation
        :return: tk.Text, the text widget
        """
        output_frame = tk.Frame(self.root)
        output_frame.pack()

        kwargs.update(
            {
                "pady": 5,
                "padx": 5,
                "fg": self.colors["disp"],
                "bg": self.colors["fg"],
                "font": self.fonts["button"],
                "width": 50,
                "height": 12,
                "wrap": tk.WORD,
            }
        )

        output_text = tk.Text(output_frame, **kwargs)
        output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(output_frame, command=output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        output_text.config(yscrollcommand=scrollbar.set)
        output_text.insert(tk.END, self.initial_text)

        return output_text

    def save_output(self) -> None:
        """
        Save the content of the output text widget to a file with a timestamp in the filename
        """
        output_text = self.output_text.get("1.0", "end-1c")
        current_datetime = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        suggested_filename = f"run_{current_datetime}.txt"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            initialfile=suggested_filename,
        )

        if file_path:
            with open(file_path, "w") as file:
                file.write(output_text)

    def show_history(self) -> None:
        """
        Display the history of runs in a separate window
        """
        history_window = Toplevel(self.root)
        history_window.title("History")
        history_window.geometry("500x500")

        history_text = tk.Text(history_window, wrap=tk.WORD, width=60, height=20)
        history_text.pack(fill=tk.BOTH, expand=True)

        for run_details in self.history_log:
            formatted_datetime = datetime.datetime.strptime(
                run_details["datetime"], "%Y-%m-%d %H:%M:%S"
            ).strftime("%d-%m-%Y %H:%M:%S")
            history_text.insert(tk.END, f"Run Date: {formatted_datetime}\n\n")
            checkbox_labels = [
                label
                for label, value in zip(self.checkbox_labels, run_details["checkboxes"])
                if value
            ]
            history_text.insert(tk.END, "Selected Checkboxes:\n")
            for label in checkbox_labels:
                history_text.insert(tk.END, f"{label}\n")
            history_text.insert(
                tk.END, f"\nOutput:\n{run_details['output']}\n{'=' * 50}\n\n"
            )

        history_text.config(state=tk.DISABLED)

    def redirect_output(self) -> None:
        """
        Redirect stdout to the Text widget for displaying output
        """
        sys.stdout = self.output_text

    def create_button(
        self, text: str, command: callable, bg: str, filepath: str, filename: str
    ) -> None:
        """
        Create a button widget with an optional icon

        :param text: str, the button text
        :param command: callable, the function to be called when the button is clicked
        :param bg: str, background color
        :param filepath: str, the path to the directory containing the image
        :param filename: str, the filename of the image
        """
        button = tk.Button(
            self.root,
            text=text,
            command=command,
            bg=bg,
            fg=self.colors["fg"],
            font=self.fonts["button"],
        )

        image_path = os.path.join(filepath, filename)
        if image_path:
            icon = Image.open(image_path)
            icon = icon.resize((15, 15), Image.LANCZOS)
            icon_image = ImageTk.PhotoImage(icon)
            button.config(image=icon_image, compound=tk.LEFT)
            button.image = icon_image

        button.pack()

    def edit_simulation_data(self) -> None:
        """
        Edit the new data row for simulation
        """
        sim_path = os.path.join(self.SU.set_sim_path(), "data.yaml")
        subprocess.Popen(["notepad.exe", sim_path])

    def show_plots(self) -> None:
        """
        Display plots from the 'plot' folder one by one
        """
        filepath = self.SU.set_plot_path()
        files = [f for f in os.listdir(filepath) if f.endswith(".png")]

        if not files:
            messagebox.showinfo("No Plots", "No plot files found in the 'plot' folder.")
            return

        self.plot_window = Toplevel(self.root)
        self.plot_window.title("Plots")
        self.plot_window.geometry("550x550")

        self.plot_images = [Image.open(os.path.join(filepath, file)) for file in files]
        self.current_plot_index = 0

        self.plot_index_label = tk.Label(
            self.plot_window,
            text=f"Plot {self.current_plot_index + 1}/{len(self.plot_images)}",
        )
        self.plot_index_label.pack()

        self.display_current_plot()
        self.plot_window.bind("<Left>", self.show_previous_plot)
        self.plot_window.bind("<Right>", self.show_next_plot)
        self.left_arrow_button = tk.Button(
            self.plot_window, text="←", command=self.show_previous_plot
        )
        self.left_arrow_button.pack(side=tk.LEFT, anchor="s")

        self.right_arrow_button = tk.Button(
            self.plot_window, text="→", command=self.show_next_plot
        )
        self.right_arrow_button.pack(side=tk.RIGHT, anchor="s")

    def display_current_plot(self) -> None:
        """
        Display the current plot in the plot viewer
        """
        if 0 <= self.current_plot_index < len(self.plot_images):
            image = self.plot_images[self.current_plot_index]
            new_size = (500, 500)
            image = image.resize(new_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            if hasattr(self, "plot_label"):
                self.plot_label.destroy()

            self.plot_label = tk.Label(self.plot_window, image=photo)
            self.plot_label.image = photo
            self.plot_label.pack()
            self.plot_index_label.config(
                text=f"Plot {self.current_plot_index + 1} out of {len(self.plot_images)}"
            )

    def show_previous_plot(self, event=None) -> None:
        """
        Show the previous plot when the left arrow key is pressed or the left arrow button is clicked
        """
        self.current_plot_index -= 1
        if self.current_plot_index < 0:
            self.current_plot_index = len(self.plot_images) - 1
        self.display_current_plot()

    def show_next_plot(self, event=None) -> None:
        """
        Show the next plot when the right arrow key is pressed or the right arrow button is clicked
        """
        self.current_plot_index += 1
        if self.current_plot_index >= len(self.plot_images):
            self.current_plot_index = 0
        self.display_current_plot()

    def run_pipeline(self) -> None:
        """
        Run the machine learning pipeline
        """
        if not any(var.get() for var in self.checkbox_vars):
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, self.nothing_text)
            return

        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, self.wait_display)
        self.output_text.update()

        data_prep = "true" if self.checkbox_vars[0].get() else "false"
        feat_eng = "true" if self.checkbox_vars[1].get() else "false"
        data_split = "true" if self.checkbox_vars[2].get() else "false"
        model_tune = "true" if self.checkbox_vars[3].get() else "false"
        model_train = "true" if self.checkbox_vars[4].get() else "false"
        model_pred = "true" if self.checkbox_vars[5].get() else "false"
        data_sql = "true" if self.checkbox_vars[6].get() else "false"

        command = [
            "python",
            os.path.join(self.config_file["project"], "audio_mnist.py"),
            "-d",
            data_prep,
            "-f",
            feat_eng,
            "-s",
            data_split,
            "-u",
            model_tune,
            "-t",
            model_train,
            "-p",
            model_pred,
            "-q",
            data_sql,
        ]

        try:
            completed_process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                check=True,
            )

            run_details = {
                "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "checkboxes": [
                    self.checkbox_vars[i].get() for i in range(len(self.checkbox_vars))
                ],
                "output": completed_process.stdout,
            }

            self.history_log.append(run_details)
            self.output_text.delete(1.0, tk.END)
            for line in completed_process.stdout.split("\n"):
                if "warnings.warn(" not in line:
                    self.output_text.insert(tk.END, line + "\n")

            self.output_text.see(tk.END)
        except subprocess.CalledProcessError as e:
            print("Error:", e)
            print("Command:", e.cmd)
            print("Exit Code:", e.returncode)
            print("Stdout:", e.stdout)
            print("Stderr:", e.stderr)
            messagebox.showerror(
                "Error", "An error occurred while running the pipeline"
            )

    def exit_app(self) -> None:
        """
        Exit the application and restore the original stdout
        """
        sys.stdout = sys.__stdout__
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
