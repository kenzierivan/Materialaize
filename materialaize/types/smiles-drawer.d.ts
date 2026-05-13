declare global {
  interface Window {
    SmilesDrawer?: {
      SvgDrawer: new (options?: {
        width?: number;
        height?: number;
        bondThickness?: number;
        atomVisualization?: string;
        padding?: number;
        themes?: Record<string, Record<string, string>>;
      }) => {
        draw(
          data: unknown,
          target: SVGElement | string | null,
          themeName?: string,
        ): void;
      };
      parse: (
        smiles: string,
        successCallback: (tree: unknown) => void,
        errorCallback?: (err: Error) => void,
      ) => void;
    };
  }
}

export {};
